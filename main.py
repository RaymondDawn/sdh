import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from utils import *
from networks import *


def main():
    if torch.cuda.is_available():
        print("---- GPU ----")
    else:
        print("---- CPU ----")

    cudnn.benchmark = True  # speed up
    assert config.DATA_DIR

    try:
        if not config.test:
            print("Making checkpoint, train_pics and val_pics dirs...")
            if not os.path.exists(config.checkpoint_save_path):
                os.makedirs(config.checkpoint_save_path)
            if not os.path.exists(config.train_pics_save_path):
                os.makedirs(config.train_pics_save_path)
            if not os.path.exists(config.val_pics_save_path):
                os.makedirs(config.val_pics_save_path)
            save_config()
        else:
            print("Making test_pics dir...")
            if not os.path.exists(config.test_pics_save_path):
                os.makedirs(config.test_pics_save_path)
    except OSError:
        print("XXXXXXXX mkdir failed XXXXXXXX")

    # process secure key
    if config.key is None or len(config.key) == 0:
        key, key_len, redundance_size = None, None, None
    else:
        print("Preprocessing secure key...")
        key = key_preprocess(config.key, config.hash_algorithm)
        key_len, redundance_size = len(key), config.key_redundance_size

    traindir = os.path.join(config.DATA_DIR, 'train')
    valdir = os.path.join(config.DATA_DIR, 'val')
    testdir = os.path.join(config.DATA_DIR, 'test')

    assert config.image_size % 32 == 0
    transform = transforms.Compose([
        transforms.Resize([config.image_size, config.image_size]),
        transforms.ToTensor()
    ])

    if not config.test:
        print("Making train and val datasets...")
        train_dataset_cover = ImageFolder(traindir, transform)
        train_dataset_secret = ImageFolder(traindir, transform)
        val_dataset_cover = ImageFolder(valdir, transform)
        val_dataset_secret = ImageFolder(valdir, transform)
    else:
        print("Making test dataset...")
        test_dataset_cover = ImageFolder(testdir, transform)
        test_dataset_secret = ImageFolder(testdir, transform)

    print("Constructing H and R networks...")
    if config.cover_dependent:
        Hnet = UnetGenerator(
            input_nc=config.channel_cover+config.channel_secret,
            output_nc=config.channel_cover,
            num_downs=config.num_downs,
            norm_type=config.norm_type,
            output_function='sigmoid',
            key_len=key_len, redundance_size=redundance_size
        )
    else:
        Hnet = UnetGenerator(
            input_nc=config.channel_secret,
            output_nc=config.channel_cover,
            num_downs=config.num_downs,
            norm_type=config.norm_type,
            output_function='tanh',
            key_len=key_len, redundance_size=redundance_size
        )
    Rnet = RevealNet(
        input_nc=config.channel_cover,
        output_nc=config.channel_secret,
        norm_type=config.norm_type,
        output_function='sigmoid',
        key_len=key_len, redundance_size=redundance_size
    )

    Hnet.apply(weights_init)
    Rnet.apply(weights_init)

    Hnet = torch.nn.DataParallel(Hnet).cuda()
    Rnet = torch.nn.DataParallel(Rnet).cuda()
    if config.checkpoint != '':
        print("Loading checkpoints for H and R...")
        checkpoint = torch.load(config.checkpoint_path)
        Hnet.load_state_dict(checkpoint['H_state_dict'])
        Rnet.load_state_dict(checkpoint['R_state_dict'])

    if config.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if config.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    if not config.test:
        print_network(Hnet)
        print_network(Rnet)

        params = list(Hnet.parameters()) + list (Rnet.parameters())
        optimizer = optim.Adam(params, lr=config.lr, betas=(0.5, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)

        print("Making train and val dataloaders...")
        train_loader_secret = DataLoader(
            train_dataset_secret,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=int(config.workers)
        )
        train_loader_cover = DataLoader(
            train_dataset_cover,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=int(config.workers)
        )
        val_loader_secret = DataLoader(
            val_dataset_secret,
            batch_size=config.batch_size,
            shuffle=False,  # do not shuffle secret image when in val mode
            num_workers=int(config.workers)
        )
        val_loader_cover = DataLoader(
            val_dataset_cover,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=int(config.workers)
        )
        
        train(
            # zip loader in train()
            train_loader_secret, train_loader_cover,
            val_loader_secret, val_loader_cover,
            Hnet, Rnet,
            optimizer, scheduler, criterion,
            cover_dependent=config.cover_dependent,
            key=key
        )
    else:
        print("Making test dataloader...")
        test_loader_secret = DataLoader(
            test_dataset_secret,
            batch_size=config.batch_size,
            shuffle=False,  # do not shuffle secret image when in test mode
            num_workers=int(config.workers)
        )
        test_loader_cover = DataLoader(
            test_dataset_cover,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=int(config.workers)
        )

        test_loader = zip(test_loader_secret, test_loader_cover)
        test(
            test_loader,
            Hnet, Rnet, criterion, config.cover_dependent,
            save_num=1, key=key, mode='test'
        )


if __name__ == '__main__':
    main()
