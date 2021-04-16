import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.options import opt
from utils.util import *
from models.networks import *


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
    transform_color = transforms.Compose([
        transforms.Resize([opt.image_size, opt.image_size]),
        transforms.ToTensor()
    ])
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([opt.image_size, opt.image_size]),
        transforms.ToTensor()
    ])
    if opt.channel_cover == 3:
        transform_cover = transform_color
    else:
        transform_cover = transform_gray
    if opt.channel_secret == 3:
        transform_secret = transform_color
    else:
        transform_secret = transform_gray

    if not opt.test:
        print("Making train and val datasets...")
        train_dataset_cover = ImageFolder(traindir, transform_cover)
        train_dataset_secret = ImageFolder(traindir, transform_secret)
        for i in range(1, opt.num_secrets):
            train_dataset_secret += ImageFolder(traindir, transform_secret)
        
        val_dataset_cover = ImageFolder(valdir, transform_cover)
        val_dataset_secret = ImageFolder(valdir, transform_secret)
        for i in range(1, opt.num_secrets):
            val_dataset_secret += ImageFolder(valdir, transform_secret)
    else:
        print("Making test dataset...")
        test_dataset_cover = ImageFolder(testdir, transform_cover)
        test_dataset_secret = ImageFolder(testdir, transform_secret)
        for i in range(1, opt.num_secrets):
            test_dataset_secret += ImageFolder(testdir, transform_secret)

    print("Constructing H and R networks...")
    if opt.cover_dependent:
        if opt.use_key:
            Hnet = UnetGenerator(
                input_nc=opt.channel_cover + opt.num_secrets * (opt.channel_secret+opt.channel_key),
                output_nc=opt.channel_cover,
                num_downs=opt.num_downs,
                norm_type=opt.norm_type,
                output_function='sigmoid'
            )
        else:
            Hnet = UnetGenerator(
                input_nc=opt.channel_cover + opt.num_secrets * opt.channel_secret,
                output_nc=opt.channel_cover,
                num_downs=opt.num_downs,
                norm_type=opt.norm_type,
                output_function='sigmoid'
            )
    else:
        if opt.use_key:
            Hnet = UnetGenerator(
                input_nc=opt.num_secrets * (opt.channel_secret + opt.channel_key),
                output_nc=opt.channel_cover,
                num_downs=opt.num_downs,
                norm_type=opt.norm_type,
                output_function='tanh'
            )
        else:
            Hnet = UnetGenerator(
            input_nc=opt.num_secrets * opt.channel_secret,
            output_nc=opt.channel_cover,
            num_downs=opt.num_downs,
            norm_type=opt.norm_type,
            output_function='tanh'
        )
    if opt.use_key:
        if opt.explicit:
            temp = opt.num_secrets
        else:
            temp = 1
        Rnet = RevealNet(
            input_nc=opt.channel_cover+opt.channel_key,
            output_nc=opt.channel_secret*temp + int(opt.explicit)*opt.channel_key*opt.num_secrets,
            norm_type=opt.norm_type,
            output_function='sigmoid'
        )
    else:
        Rnet = RevealNet(
        input_nc=opt.channel_cover,
        output_nc=opt.channel_secret,
        norm_type=opt.norm_type,
        output_function='sigmoid'
    )

    Enet = EncodingNet(opt.image_size, opt.channel_key, opt.redundance, opt.batch_size)
    Enet = torch.nn.DataParallel(Enet).cuda()

    Hnet.apply(weights_init)
    Rnet.apply(weights_init)
    Hnet = torch.nn.DataParallel(Hnet).cuda()
    Rnet = torch.nn.DataParallel(Rnet).cuda()

    if opt.adversary:
        Adversary = AdversarialNet(input_nc=opt.channel_cover)
        Adversary = torch.nn.DataParallel(Adversary).cuda()
    else:
        Adversary, optimizer_adv = None, None

    if opt.load_checkpoint:
        print("Loading checkpoints for networks...")
        checkpoint = torch.load(opt.checkpoint_path)
        Hnet.load_state_dict(checkpoint['H_state_dict'])
        Rnet.load_state_dict(checkpoint['R_state_dict'])
        if opt.adversary:
            Adversary.load_state_dict(checkpoint['Adversary_state_dict'])
        if opt.redundance != -1:
            Enet.load_state_dict(checkpoint['E_state_dict'])

    NoiseLayers = torch.nn.DataParallel(AttackNet(noise_type=opt.noise_type)).cuda()

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    if not opt.test:
        print_network(Hnet)
        print_network(Rnet)

        params = list(Hnet.parameters()) + list (Rnet.parameters())
        if opt.adversary:
            print_network(Adversary)
            optimizer_adv = optim.Adam(Adversary.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        if opt.redundance != -1:
            print_network(Enet)
            params += list(Enet.parameters())
        optimizer = optim.Adam(params, lr=opt.lr, betas=(0.5, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)

        print("Making train and val dataloaders...")
        train_loader_secret = DataLoader(
            train_dataset_secret,
            batch_size=opt.batch_size*opt.num_secrets,
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
            batch_size=opt.batch_size*opt.num_secrets,
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
            Hnet, Rnet, Enet, NoiseLayers,
            optimizer, scheduler, criterion,
            opt.cover_dependent, opt.use_key, Adversary, optimizer_adv
        )
    else:
        print("Making test dataloader...")
        test_loader_secret = DataLoader(
            test_dataset_secret,
            batch_size=opt.batch_size*opt.num_secrets,
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
            test_loader, Hnet, Rnet, Enet, NoiseLayers,
            criterion, opt.cover_dependent, opt.use_key,
            save_num=1, mode='test', epoch=None
        )


if __name__ == '__main__':
    main()
