import os
from PIL import Image
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.options import opt
from utils.util import *
from models import *


def multi_hiding(n=2):
    cudnn.benchmark = True  # speed up

    print("Making analysis dir...")
    try:
        if not os.path.exists(opt.analysis_pics_save_dir):
            os.makedirs(opt.analysis_pics_save_dir)
    except OSError:
        print("XXXXXXXX mkdir failed XXXXXXXX")

    valdir = os.path.join(opt.data_dir, 'val')
    testdir = os.path.join(opt.data_dir, 'test')

    assert opt.image_size % 32 == 0
    transform = transforms.Compose([
        transforms.Resize([opt.image_size, opt.image_size]),
        transforms.ToTensor()
    ])
    
    print("Making datasets and dataloaders...")
    dataset_cover = ImageFolder(valdir, transform)
    loader_cover = DataLoader(
        dataset_cover,
        batch_size=opt.batch_size,
        shuffle=True
    )
    dataset_secret = ImageFolder(testdir, transform)
    loader_secret = DataLoader(
        dataset_secret,
        batch_size=opt.batch_size*n,  # hiding n secrets into 1 cover
        shuffle=True
    )

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

    print("Hiding secrets...")
    for i, (secret, cover) in enumerate(zip(loader_secret, loader_cover), start=1):
        key_cache, secret_split_cache = [], []
        b, c, h, w = cover.shape
        secret, cover = secret.cuda(), cover.cuda()
        container = cover  # initialization
        for j in range(n):
            secret_split = secret[b*j:b*(j+1), :, :, :]
            secret_split_cache.append(secret_split)

            key = torch.Tensor([float(torch.randn(1)<0) for _ in range(w)]).cuda()
            key_cache.append(key)
            red_key = key.view(1, 1, 1, w).repeat(b, c, h, 1)

            H_input = torch.cat((secret_split, red_key), dim=1)
            H_output = Hnet(H_input)

            container = H_output + container
        
        cover_gap = ((container - cover)*10 + 0.5).clamp_(0.0, 1.0)
        show_all = torch.cat((cover, container, cover_gap), dim=0)
        for j in range(n):
            red_key = key_cache[j].view(1, 1, 1, w).repeat(b, c, h, 1)
            rev_secret = Rnet(torch.cat((container, red_key), dim=1))
            show_all = torch.cat((show_all, secret_split_cache[j], rev_secret), dim=0)
        
        fake_key = torch.Tensor([float(torch.randn(1)<0) for _ in range(w)]).cuda().view(1, 1, 1, w).repeat(b, c, h, 1)
        fake_secret = Rnet(torch.cat((container, fake_key), dim=1))
        show_all = torch.cat((show_all, fake_secret), dim=0)

        save_path = '%s/hiding_%02d_secrets.png' % (opt.analysis_pics_save_dir, n)
        grid = vutils.make_grid(show_all, nrow=b, padding=1, normalize=False)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(save_path)

        break


if __name__ == '__main__':
    multi_hiding()
