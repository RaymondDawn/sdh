import os
import copy
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append(".")

from utils.options import opt
from utils.util import *
from models.networks import *


def main(fake=True):
    if torch.cuda.is_available():
        print("---- GPU ----")
    else:
        print("---- CPU ----")

    cudnn.benchmark = True  # speed up
    assert opt.data_dir

    try:
        print("Making analysis dir...")
        if not os.path.exists(opt.analysis_pics_save_dir):
            os.makedirs(opt.analysis_pics_save_dir)
    except OSError:
        print("XXXXXXXX mkdir failed XXXXXXXX")

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
    
    print("Making datasets and dataloaders...")
    dataset_cover = ImageFolder(testdir, transform_cover)
    loader_cover = DataLoader(
        dataset_cover,
        batch_size=opt.batch_size,
        shuffle=True
    )
    dataset_secret = ImageFolder(testdir, transform_secret)
    for i in range(1, opt.num_secrets):
        dataset_secret += ImageFolder(testdir, transform_secret)
    loader_secret = DataLoader(
        dataset_secret,
        batch_size=opt.batch_size*opt.num_secrets,
        shuffle=False
    )

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
            output_nc=opt.channel_secret*temp + int(opt.explicit)*opt.channel_prob*opt.num_secrets,
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
    if opt.load_checkpoint:
        print("Loading checkpoints for H and R...")
        checkpoint = torch.load(opt.checkpoint_path)
        Hnet.load_state_dict(checkpoint['H_state_dict'])
        Rnet.load_state_dict(checkpoint['R_state_dict'])
        if opt.redundance != -1:
            Enet.load_state_dict(checkpoint['E_state_dict'])

    NoiseLayers = torch.nn.DataParallel(AttackNet(noise_type=opt.noise_type)).cuda()

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    # turn on val mode
    Hnet.eval()
    Rnet.eval()

    for i, (secret, cover) in enumerate(zip(loader_secret, loader_cover), start=1):
        cover, secret = cover.cuda(), secret.cuda()
        (b, c, h, w), (_, c_s, h_s, w_s) = cover.shape, secret.shape
        assert h == h_s and w == w_s and opt.num_secrets == 1

        key = torch.Tensor([float(torch.randn(1)<0) for _ in range(w)]).cuda()
        red_key = Enet(key)

        fake_key, s = key.clone(), set()
        for j in range(64):
            index = (j + int(np.random.rand() * 128)) % 128
            while index in s:
                index = (j + int(np.random.rand() * 128)) % 128
            s.add(index)
            fake_key[index] = -fake_key[index] + 1  # 0->1; 1->0
        red_fake_key = Enet(fake_key)

        H_input = torch.cat((secret, red_key), dim=1)
        H_output = Hnet(H_input)
        container = H_output + cover

        assert opt.feature_map
        if fake:
          X1, X2, X3, X4, X5, output = Rnet(torch.cat((container, red_fake_key), dim=1))
        else:
          X1, X2, X3, X4, X5, output = Rnet(torch.cat((container, red_fake_key), dim=1))

        X = X5
        d, index = X.shape[1], 22
        for j in range(d//64):
          for k in range(64):
              ax = plt.subplot(8, 8, k + 1)
              ax.axis('off')
              plt.imshow(X.detach().cpu().numpy()[index,k,:,:])
          plt.savefig('%s/feature_map_%02d.png' % (opt.analysis_pics_save_dir, j))
          plt.show()

        plt.imshow(container.permute(0,2,3,1).detach().cpu().numpy()[index,:,:,:])
        ax.axis('off')
        plt.savefig('%s/container.png' % opt.analysis_pics_save_dir)
        plt.show()
        plt.imshow(secret.permute(0,2,3,1).detach().cpu().numpy()[index,:,:,:])
        ax.axis('off')
        plt.savefig('%s/secret.png' % opt.analysis_pics_save_dir)
        plt.show()
        plt.imshow(output.permute(0,2,3,1).detach().cpu().numpy()[index,:,:,:])
        ax.axis('off')
        plt.savefig('%s/output.png' % opt.analysis_pics_save_dir)
        plt.show()
        
        break


if __name__ == '__main__':
    main(fake=True)
