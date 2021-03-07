import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.options import opt
from utils.util import *
from models import *


def main():
    if torch.cuda.is_available():
        print("---- GPU ----")
    else:
        print("---- CPU ----")

    cudnn.benchmark = True  # speed up
    assert opt.data_dir

    try:
        if not opt.test:
            print("Making checkpoint, train_pics and val_pics dirs...")
            if not os.path.exists(opt.checkpoints_save_dir):
                os.makedirs(opt.checkpoints_save_dir)
            if not os.path.exists(opt.train_pics_save_dir):
                os.makedirs(opt.train_pics_save_dir)
            if not os.path.exists(opt.val_pics_save_dir):
                os.makedirs(opt.val_pics_save_dir)
            save_options()
        else:
            print("Making test_pics dir...")
            if not os.path.exists(opt.test_pics_save_dir):
                os.makedirs(opt.test_pics_save_dir)
    except OSError:
        print("XXXXXXXX mkdir failed XXXXXXXX")

    traindir = os.path.join(opt.data_dir, 'train')
    valdir = os.path.join(opt.data_dir, 'val')
    testdir = os.path.join(opt.data_dir, 'test')

    assert opt.image_size % 32 == 0
    transform = transforms.Compose([
        transforms.Resize([opt.image_size, opt.image_size]),
        transforms.ToTensor()
    ])

    if not opt.test:
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
    if opt.cover_dependent:
        Hnet = UnetGenerator(
            input_nc=opt.channel_cover+opt.channel_secret+opt.channel_secret,
            output_nc=opt.channel_cover,
            num_downs=opt.num_downs,
            norm_type=opt.norm_type,
            output_function='sigmoid'
        )
    else:
        Hnet = UnetGenerator(
            input_nc=opt.channel_secret+opt.channel_secret,
            output_nc=opt.channel_cover,
            num_downs=opt.num_downs,
            norm_type=opt.norm_type,
            output_function='tanh'
        )
    Rnet = RevealNet(
        input_nc=opt.channel_cover+opt.channel_secret,
        output_nc=opt.channel_secret,
        norm_type=opt.norm_type,
        output_function='sigmoid'
    )

    Hnet.apply(weights_init)
    Rnet.apply(weights_init)

    Hnet = torch.nn.DataParallel(Hnet).cuda()
    Rnet = torch.nn.DataParallel(Rnet).cuda()
    if opt.load_checkpoint:
        print("Loading checkpoints for H and R...")
        checkpoint = torch.load(opt.checkpoint_path)
        Hnet.load_state_dict(checkpoint['H_state_dict'])
        Rnet.load_state_dict(checkpoint['R_state_dict'])

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    if not opt.test:
        print_network(Hnet)
        print_network(Rnet)

        params = list(Hnet.parameters()) + list (Rnet.parameters())
        optimizer = optim.Adam(params, lr=opt.lr, betas=(0.5, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)

        print("Making train and val dataloaders...")
        train_loader_secret = DataLoader(
            train_dataset_secret,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers
        )
        train_loader_cover = DataLoader(
            train_dataset_cover,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers
        )
        val_loader_secret = DataLoader(
            val_dataset_secret,
            batch_size=opt.batch_size,
            shuffle=False,  # do not shuffle secret image when in val mode
            num_workers=opt.workers
        )
        val_loader_cover = DataLoader(
            val_dataset_cover,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers
        )
        
        train(
            # zip loader in train()
            train_loader_secret, train_loader_cover,
            val_loader_secret, val_loader_cover,
            Hnet, Rnet,
            optimizer, scheduler, criterion,
            cover_dependent=opt.cover_dependent
        )
    else:
        print("Making test dataloader...")
        test_loader_secret = DataLoader(
            test_dataset_secret,
            batch_size=opt.batch_size,
            shuffle=False,  # do not shuffle secret image when in test mode
            num_workers=opt.workers
        )
        test_loader_cover = DataLoader(
            test_dataset_cover,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers
        )

        test_loader = zip(test_loader_secret, test_loader_cover)
        inference(
            test_loader, Hnet, Rnet,
            criterion, opt.cover_dependent, save_num=1
        )


if __name__ == '__main__':
    main()
