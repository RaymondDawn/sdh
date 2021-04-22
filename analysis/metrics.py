import os
import copy
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from lpips import LPIPS

import sys
sys.path.append(".")

from utils.options import opt
from utils.util import *
from models.networks import *


def main(num_saves=1, partial=True):
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

    # metrics
    batch_size = opt.batch_size
    H_APD  , R_APD   = AverageMeter(), AverageMeter()
    H_PSNR , R_PSNR  = AverageMeter(), AverageMeter()
    H_SSIM , R_SSIM  = AverageMeter(), AverageMeter()
    H_LPIPS, R_LPIPS = AverageMeter(), AverageMeter()
    R_APD_, R_APD_s  = AverageMeter(), AverageMeter()
    Count = AverageMeter()

    # turn on val mode
    Hnet.eval()
    Rnet.eval()

    loss_fn_alex = LPIPS(net='alex')
    print("Hiding secrets and calculating metrics...")
    for i, (secret, cover) in tqdm(enumerate(zip(loader_secret, loader_cover), start=1)):
        cover, container, secret_set, rev_secret_set, rev_secret_, H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff \
            = forward_pass(secret, cover, Hnet, Rnet, Enet, NoiseLayers, criterion, opt.cover_dependent, opt.use_key, None)

        cover, container = cover.detach().cpu(), container.detach().cpu()
        for j in range(opt.num_secrets):
            secret_set[j] = secret_set[j].detach().cpu()
            rev_secret_set[j] = rev_secret_set[j].detach().cpu()
        if opt.use_key:
            rev_secret_ = rev_secret_.detach().cpu()

        H_APD.update(H_diff.item(), batch_size)
        R_APD.update(R_diff.item(), batch_size)
        R_APD_s.update((rev_secret_set[0]).abs().mean() * 255, batch_size)
        h_psnr, r_psnr = PSNR(cover, container), 0
        h_ssim, r_ssim = SSIM(cover, container), 0
        h_lpips, r_lpips = loss_fn_alex(cover, container).mean(), 0
        for j in range(opt.num_secrets):
            r_psnr += PSNR(secret_set[j], rev_secret_set[j])
            r_ssim += SSIM(secret_set[j], rev_secret_set[j])
            r_lpips += loss_fn_alex(secret_set[j], rev_secret_set[j]).mean()
        H_PSNR.update(h_psnr, batch_size)
        H_SSIM.update(h_ssim, batch_size)
        H_LPIPS.update(h_lpips, batch_size)
        R_PSNR.update(r_psnr/opt.num_secrets, batch_size)
        R_SSIM.update(r_ssim/opt.num_secrets, batch_size)
        R_LPIPS.update(r_lpips/opt.num_secrets, batch_size)
        if opt.use_key and not opt.explicit:
            R_APD_.update(R_diff_.item(), batch_size)
        Count.update(count_diff, 1)

        cover_gap = ((container-cover)*10 + 0.5).clamp_(0.0, 1.0)
        show_all = torch.cat((cover, container, cover_gap), dim=0)
        for j in range(opt.num_secrets):
            temp_s = secret_set[j].repeat(1, 3//opt.channel_secret, 1, 1)
            temp_rev_s = rev_secret_set[j].repeat(1, 3//opt.channel_secret, 1, 1)
            show_all = torch.cat((show_all, temp_s, temp_rev_s), dim=0)
        if opt.num_secrets == 1:
            secret_gap = ((temp_rev_s-temp_s)*10 + 0.5).clamp_(0.0, 1.0)
            show_all = torch.cat((show_all, secret_gap), dim=0)
        if opt.use_key:
            if not opt.explicit:
                show_all = torch.cat((show_all, (rev_secret_*50).repeat(1, 3//opt.channel_secret, 1, 1)), dim=0)
            else:
                show_all = torch.cat((show_all, rev_secret_.repeat(1, 3//opt.channel_secret, 1, 1)), dim=0)

        #temp = rev_secret_set[0]*0.5 + rev_secret_set[1]*0.5
        #show_all = torch.cat((show_all, temp), dim=0)

        if i <= num_saves:
            save_path = '%s/hiding_%02d_secrets_modified_%03d_bits%d.png' % (opt.analysis_pics_save_dir, opt.num_secrets, opt.modified_bits, i)
            grid = vutils.make_grid(show_all, nrow=opt.batch_size, padding=1, normalize=False)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(save_path)
        if partial and i == 20:
            break

    log  = '\nH_APD=%.4f\tH_PSNR=%.4f\tH_SSIM=%.4f\tH_LPIPS=%.4f\nR_APD=%.4f\tR_PSNR=%.4f\tR_SSIM=%.4f\tR_LPIPS=%.4f\tR_APD_=%.4f\tR_APD_s=%.4f\tCount=%.4f' % (
        H_APD.avg, H_PSNR.avg, H_SSIM.avg, H_LPIPS.avg,
        R_APD.avg, R_PSNR.avg, R_SSIM.avg, R_LPIPS.avg,
        R_APD_.avg, R_APD_s.avg, Count.avg
    )
    print_log(log)


if __name__ == '__main__':
    main(num_saves=3, partial=True)
