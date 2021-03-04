import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision import transforms

from lpips import LPIPS

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
        print("Makeing analysis dir...")
        analdir = config.experiment_dir + '/analysis/%s' % config.checkpoint_mode
        if not os.path.exists(analdir):
            os.makedirs(analdir)
        save_config()
    except OSError:
        print("XXXXXXXX mkdir failed XXXXXXXX")

    # process secure key
    if config.key is None or len(config.key) == 0:
        key, key_len, redundance_size = None, None, None
    else:
        print("Preprocessing secure key...")
        key = key_preprocess(config.key, config.hash_algorithm)
        key_len, redundance_size = len(key), config.key_redundance_size

    testdir = os.path.join(config.DATA_DIR, 'test')

    assert config.image_size % 32 == 0
    transform = transforms.Compose([
        transforms.Resize([config.image_size, config.image_size]),
        transforms.ToTensor()
    ])

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
    if config.noise:
        Anet = AttackNet(noise_type=config.noise_type)
    else:
        Anet = Identity()

    Hnet.apply(weights_init)
    Rnet.apply(weights_init)

    Hnet = torch.nn.DataParallel(Hnet).cuda()
    Rnet = torch.nn.DataParallel(Rnet).cuda()
    Anet = torch.nn.DataParallel(Anet).cuda()
    if config.checkpoint != '':
        print("Loading checkpoints for H and R...")
        checkpoint = torch.load(config.checkpoint_path)
        Hnet.load_state_dict(checkpoint['H_state_dict'])
        Rnet.load_state_dict(checkpoint['R_state_dict'])
    else:
        raise RuntimeError('checkpoints are expected to be loaded in `analysis` mode')

    if config.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if config.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    print_network(Hnet)
    print_network(Rnet)
    print_network(Anet)

    print("Making test dataloader...")
    test_loader_secret = DataLoader(
        test_dataset_secret,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=int(config.workers)
    )
    test_loader_cover = DataLoader(
        test_dataset_cover,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=int(config.workers)
    )
    test_loader = zip(test_loader_secret, test_loader_cover)

    print("\n#### analysis begin ####")
    batch_size = config.batch_size
    # analysis information
    H_APD  , R_APD   = AverageMeter(), AverageMeter()
    H_PSNR , R_PSNR  = AverageMeter(), AverageMeter()
    H_SSIM , R_SSIM  = AverageMeter(), AverageMeter()
    H_LPIPS, R_LPIPS = AverageMeter(), AverageMeter()
    R_APD_ = AverageMeter()

    # turn on val mode
    Hnet.eval()
    Rnet.eval()

    loss_fn = LPIPS(net='alex')
    for i, (secret_image, cover_image) in tqdm(enumerate(test_loader, start=1)):
        cover_image, container_image, secret_image, rev_secret_image, rev_secret_image_, _, _, _, H_diff, R_diff, R_diff_ \
                = forward_pass(secret_image, cover_image, Hnet, Rnet, Anet, criterion, config.cover_dependent, key)
        H_APD.update(H_diff.item(), batch_size)
        R_APD.update(R_diff.item(), batch_size)

        cover_image, container_image = cover_image.detach().cpu(), container_image.detach().cpu()
        secret_image, rev_secret_image = secret_image.detach().cpu(), rev_secret_image.detach().cpu()
        if key is not None:
            rev_secret_image_ = rev_secret_image_.detach().cpu()

        h_psnr = PSNR(cover_image, container_image)
        r_psnr = PSNR(secret_image, rev_secret_image)
        h_ssim = SSIM(cover_image, container_image)
        r_ssim = SSIM(secret_image, rev_secret_image)
        h_lpips = loss_fn.forward(cover_image, container_image).mean()
        r_lpips = loss_fn.forward(secret_image, rev_secret_image).mean()

        H_PSNR.update(h_psnr, batch_size)
        R_PSNR.update(r_psnr, batch_size)
        H_SSIM.update(h_ssim, batch_size)
        R_SSIM.update(r_ssim, batch_size)
        H_LPIPS.update(h_lpips, batch_size)
        R_LPIPS.update(r_lpips, batch_size)

        if key is not None:
            R_APD_.update(R_diff_.item(), batch_size)
        
        if i == 1:
            save_result_pic(
                batch_size,
                cover_image, container_image,
                secret_image, rev_secret_image, rev_secret_image_,
                epoch=None, i=i,  # epoch is None in test mode
                save_path=config.anal_pics_save_path
            )

        if i <= 10:
            save_image(cover_image, config.anal_pics_save_path + '/covers/%d.png' % i)
            save_image(container_image, config.anal_pics_save_path + '/containers/%d.png' % i)
            save_image(secret_image, config.anal_pics_save_path + '/secrets/%d.png' % i)
            save_image(rev_secret_image, config.anal_pics_save_path + '/rev_secrets/%d.png' % i)

    log  = '\nH_APD=%.4f\tH_PSNR=%.4f\tH_SSIM=%.4f\tH_LPIPS=%.4f\nR_APD=%.4f\tR_PSNR=%.4f\tR_SSIM=%.4f\tR_LPIPS=%.4f\tR_APD_=%.4f' % (
        H_APD.avg, H_PSNR.avg, H_SSIM.avg, H_LPIPS.avg,
        R_APD.avg, R_PSNR.avg, R_SSIM.avg, R_LPIPS.avg,
        R_APD_.avg
    )
    
    print_log(log)
    print("#### analysis end ####\n")


if __name__ == '__main__':
    main()
