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
    cudnn.benchmark = True  # speed up
    assert config.DATA_DIR

    try:
        if config.test_path == '':
            if not os.path.exists(config.checkpoint_path):
                os.makedirs(config.checkpoint_path)
            if not os.path.exists(config.train_pics_save_path):
                os.makedirs(config.train_pics_save_path)
            if not os.path.exists(config.val_pics_save_path):
                os.makedirs(config.val_pics_save_path)
            if not os.path.exists(config.log_path):
                os.makedirs(config.log_path)
        else:
            pass
    except OSError:
        print("XXXXXXXX mkdir failed XXXXXXXX")

    # save config
    save_config()

    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')

    assert config.image_size % 32 == 0
    transform = transforms.Compose([
        transforms.Resize([config.image_size, config.image_size]),
        transforms.ToTensor()
    ])

    if config.test_path == '':
        train_dataset_cover = ImageFolder(traindir, transform)
        train_dataset_secret = ImageFolder(traindir, transform)
        val_dataset_cover = ImageFolder(valdir, transform)
        val_dataset_secret = ImageFolder(valdir, transform)
    else:
        pass

    if config.cover_dependent:
        Hnet = UnetGenerator(
            input_nc=config.channel_secret,
            output_nc=config.channel_cover,
            num_downs=config.num_downs,
            norm_type=config.norm_type,
            output_function='sigmoid'
        )
    else:
        Hnet = UnetGenerator(
            input_nc=config.channel_secret,
            output_nc=config.channel_cover,
            num_downs=config.num_downs,
            norm_type=config.norm_type,
            output_function='tanh'
        )
    Rnet = RevealNet(
        input_nc=config.channel_cover,
        output_nc=config.channel_secret,
        norm_type=config.norm_type,
        output_function='sigmoid'
    )

    Hnet.apply(weights_init)
    Rnet.apply(weights_init)

    Hnet = torch.nn.DataParallel(Hnet).cuda()
    Rnet = torch.nn.DataParallel(Rnet).cuda()
    if config.checkpoint != "":
        if config.checkpoint_diff == "":
            checkpoint = torch.load(config.checkpoint)
            Hnet.load_state_dict(checkpoint['H_state_dict'])
            Rnet.load_state_dict(checkpoint['R_state_dict'])
        else:
            checkpoint = torch.load(config.checkpoint)
            checkpoint_diff = torch.load(config.checkpoint_diff)
            Hnet.load_state_dict(checkpoint_diff['H_state_dict'])
            Rnet.load_state_dict(checkpoint['R_state_dict'])
    
    print_network(Hnet)
    print_network(Rnet)

    if config.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if config.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    if config.test_path == '':
        params = list(Hnet.parameters()) + list (Rnet.parameters())
        optimizer = optim.Adam(params, lr=config.lr, betas=(0.5, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)

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

        train_loader = zip(train_loader_secret, train_loader_cover)
        val_loader = zip(val_loader_secret, val_loader_cover)
        train(
            train_loader, val_loader,
            Hnet, Rnet,
            optimizer, scheduler, criterion,
            cover_dependent=config.cover_dependent
        )
    else:
        pass


if __name__ == '__main__':
    main()
