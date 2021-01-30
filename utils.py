import os
import time
import shutil
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.utils as vutils

import config
from image_folder import ImageFolder


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def key_preprocess(key: str, algorithm='md5') -> torch.Tensor:
    """Hash and binarize the key."""
    if algorithm == 'md5':
        hash_key = hashlib.md5(key.encode(encoding='UTF-8')).digest()
    elif algorithm == 'sha256':
        hash_key = hashlib.sha256(key.encode(encoding='UTF-8')).digest()
    elif algorithm == 'sha512':
        hash_key = hashlib.sha512(key.encode(encoding='UTF-8')).digest()
    else:
        raise NotImplementedError('hash algorithm [%s] is not found' % algorithm)
    binary_key = ''.join(format(x, '08b') for x in hash_key)
    tensor_key = torch.Tensor([float(x)/2 + 0.25 for x in binary_key])  # [0, 1] -> [0.25, 0.75]
    
    return tensor_key


def weights_init(m):
    """Weights initialization for a network."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out') 
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


def print_log(log_info, log_path=config.log_path, console=True, debug=False):
    """Print log information to the console and log files."""
    if console:  # print the info into the console
        print(log_info)
    if not debug:  # debug mode don't write the log into files
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
            fp.close()
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


def print_network(net, log_path=config.log_path):
    """Print network information."""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), log_path)
    print_log('Total number of parameters: %d\n' % num_params)


def save_config():
    """Save configuations as .txt file."""
    fp = open(config.config_path, "w")
    fp.writelines("ngpu\t\t\t\t%d\n" % config.ngpu)
    fp.writelines("workers\t\t\t\t%d\n" % config.workers)
    fp.writelines("image_size\t\t\t\t%d\n" % config.image_size)
    fp.writelines("training_dataset_size\t\t\t\t%d\n" % config.training_dataset_size)
    fp.writelines("exper_name\t\t\t\t%s\n" % config.exper_name)
    fp.writelines("ROOT\t\t\t\t%s\n" % config.ROOT)
    fp.writelines("DATA_DIR\t\t\t\t%s\n" % config.DATA_DIR)
    fp.writelines("experiment_dir\t\t\t\t%s\n" % config.experiment_dir)
    fp.writelines("config_path\t\t\t\t%s\n" % config.config_path)
    fp.writelines("log_path\t\t\t\t%s\n" % config.log_path)
    fp.writelines("checkpoint_save_path\t\t\t\t%s\n" % config.checkpoint_save_path)
    fp.writelines("train_pics_save_path\t\t\t\t%s\n" % config.train_pics_save_path)
    fp.writelines("train_loss_save_path\t\t\t\t%s\n" % config.train_loss_save_path)
    fp.writelines("val_pics_save_path\t\t\t\t%s\n" % config.val_pics_save_path)
    fp.writelines("test_pics_save_path\t\t\t\t%s\n" % config.test_pics_save_path)
    fp.writelines("checkpoint\t\t\t\t%s\n" % config.checkpoint)
    fp.writelines("checkpoint_path\t\t\t\t%s\n" % config.checkpoint_path)
    fp.writelines("test\t\t\t\t%s\n" % config.test)
    fp.writelines("epochs\t\t\t\t%d\n" % config.epochs)
    fp.writelines("batch_size\t\t\t\t%d\n" % config.batch_size)
    fp.writelines("beta\t\t\t\t%f\n" % config.beta)
    fp.writelines("gamma\t\t\t\t%f\n" % config.gamma)
    fp.writelines("lr\t\t\t\t%f\n" % config.lr)
    fp.writelines("lr_decay_freq\t\t\t\t%d\n" % config.lr_decay_freq)
    fp.writelines("iters_per_epoch\t\t\t\t%d\n" % config.iters_per_epoch)
    fp.writelines("log_freq\t\t\t\t%d\n" % config.log_freq)
    fp.writelines("result_pic_freq\t\t\t\t%d\n" % config.result_pic_freq)
    fp.writelines("key\t\t\t\t%s\n" % config.key)
    fp.writelines("hash_algorithm\t\t\t\t%s\n" % config.hash_algorithm)
    fp.writelines("key_redundance_size\t\t\t\t%s\n" % config.key_redundance_size)
    fp.writelines("cover_dependent\t\t\t\t%s\n" % config.cover_dependent)
    fp.writelines("channel_secret\t\t\t\t%d\n" % config.channel_secret)
    fp.writelines("channel_cover\t\t\t\t%d\n" % config.channel_cover)
    fp.writelines("num_downs\t\t\t\t%d\n" % config.num_downs)
    fp.writelines("norm_type\t\t\t\t%s\n" % config.norm_type)
    fp.writelines("loss\t\t\t\t%s\n" % config.loss)
    fp.close()


def save_checkpoint(state, is_best):
    """Save checkpoint files for training."""
    if is_best:  # best
        filename = '%s/checkpoint_best.pth.tar' % (config.checkpoint_save_path)
    else:  # newest
        filename = '%s/checkpoint_newest.pth.tar' % (config.checkpoint_save_path)
    torch.save(state, filename)


def save_image(input_image, image_path):
    """Save a batch torch.Tensor as an image to the disk."""
    if isinstance(input_image, torch.Tensor):  # detach the tensor from current graph
        image_tensor = input_image.detach()
    else:
        raise TypeError("Type of the input is neither `np.ndarray` nor `torch.Tensor`")
    image_numpy = image_tensor[0].cpu().float().numpy()  # .numpy() will cause deviation on pixels  e.g. tensor(-0.5059) -> array(0.5058824)
    image_numpy = np.round(np.transpose(image_numpy, (1, 2, 0)) * 255.0)  # [c, h, w] -> [h,w,c] & [0,1] -> [0,255]

    image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    image_pil.save(image_path)


def save_result_pic(batch_size, cover, container, secret, rev_secret, rev_secret_, epoch, i, save_path):
    """Save a batch of result pictures."""
    if epoch is None:
        result_name = '%s/result_pic_batch%04d.png' % (save_path, i)
    else:
        result_name = '%s/result_pic_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    cover_gap = container - cover
    secret_gap = rev_secret - secret
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)
    # cover_gap = (container - cover).abs() * 10
    # secret_gap = (rev_secret - secret).abs() * 10

    show_cover = torch.cat((cover, container, cover_gap), dim=0)
    show_secret = torch.cat((secret, rev_secret, secret_gap), dim=0)
    if rev_secret_ is None:
        show_all = torch.cat((show_cover, show_secret), dim=0)
    else:
        show_all = torch.cat((show_cover, show_secret, rev_secret_), dim=0)

    vutils.save_image(show_all, result_name, batch_size, padding=1, normalize=True)
    # vutils.save_image(show_all, result_name, batch_size, padding=1, normalize=False)


def save_loss_pic(h_losses_list, r_losses_list, r_losses_list_, save_path):
    """Save loss picture for Hnet and Rnet."""
    plt.title('Training Loss for H and R')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.plot(list(range(1, len(h_losses_list)+1)), h_losses_list, label='H loss')
    plt.plot(list(range(1, len(r_losses_list)+1)), r_losses_list, label='R loss')
    if len(r_losses_list_) != 0:
        plt.plot(list(range(1, len(r_losses_list_)+1)), r_losses_list_, label='R loss (fake)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def adjust_learning_rate(optimizer, epoch):
    """Set the learning rate to the initial LR decayed by 10 every `lr_decay_freq` epochs."""
    lr = config.lr * (0.1 ** (epoch // config.lr_decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def forward_pass(secret_image, cover_image, Hnet, Rnet, Anet, criterion, cover_dependent, key):
    """Forward propagation for hiding and reveal network and calculate losses and APD.
    
    Parameters:
        secret_image (torch.Tensor) -- secret images in batch
        cover_image (torch.Tensor)  -- cover images in batch
        Hnet (nn.Module)            -- hiding network
        Rnet (nn.Module)            -- reveal network
        Anet (nn.Module)            -- attack network, i.e. noise layers
        criterion                   -- loss function
        cover_dependent (bool)      -- DDH (dependent deep hiding) or UDH (universal deep hiding)
        key (torch.Tensor)          -- secure key (`None` denotes no key)
    """
    cover_image = cover_image.cuda()
    secret_image = secret_image.cuda()
    if key is not None:
        key = key.cuda()

    if cover_dependent:
        H_input = torch.cat((cover_image, secret_image), dim=1)
    else:
        H_input = secret_image

    H_output = Hnet(H_input, key)

    if cover_dependent:
        container_image = H_output
    else:
        container_image = H_output + cover_image

    H_loss = criterion(container_image, cover_image)

    container_image = Anet(container_image)

    rev_secret_image = Rnet(container_image, key)
    R_loss = criterion(rev_secret_image, secret_image)

    if key is None:
        rev_secret_image_, R_loss_, R_diff_ = None, 0, 0
    else:
        fake_key = torch.Tensor([float(torch.randn(1)<0)/2 + 0.25 for _ in range(len(key))]).cuda()  # binary
        rev_secret_image_ = Rnet(container_image, fake_key)
        R_loss_ = criterion(rev_secret_image_, torch.zeros(rev_secret_image_.size(), device=rev_secret_image_.device))
        R_diff_ = (rev_secret_image_).abs().mean() * 255

    # L1 metric (APD: average pixel difference)
    H_diff = (container_image - cover_image).abs().mean() * 255
    R_diff = (rev_secret_image - secret_image).abs().mean() * 255

    return cover_image, container_image, secret_image, rev_secret_image, rev_secret_image_, H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_


def train(train_loader_secret, train_loader_cover, val_loader_secret, val_loader_cover, Hnet, Rnet, Anet, optimizer, scheduler, criterion, cover_dependent, key):
    """Train Hnet and Rnet and schedule learning rate by the validation results.
    
    Parameters:
        train_loader_secret     -- train_loader for secret images
        train_loader_cover      -- train_loader for cover images
        val_loader_secret       -- val_loader for secret images
        val_loader_cover        -- val_loader for cover images
        Hnet (nn.Module)        -- hiding network
        Rnet (nn.Module)        -- reveal network
        Anet (nn.Module)        -- attack network, i.e. noise layers
        optimizer               -- optimizer for Hnet and Rnet
        scheduler               -- scheduler for optimizer to set dynamic learning rate
        criterion               -- loss function
        cover_dependent (bool)  -- DDH (dependent deep hiding) or UDH (universal deep hiding)
        key (torch.Tensor)      -- secure key (`None` denotes no key)
    """
    #### training and update parameters ####
    MIN_LOSS = float('inf')
    h_losses_list, r_losses_list, r_losses_list_ = [], [], []
    print("######## TRAIN BEGIN ########")
    for epoch in range(config.epochs):
        adjust_learning_rate(optimizer, epoch)
        # must zip in epoch's iteration
        train_loader = zip(train_loader_secret, train_loader_cover)
        val_loader = zip(val_loader_secret, val_loader_cover)

        # training information
        batch_time = AverageMeter()  # time for processing a batch
        data_time = AverageMeter()   # time for reading data
        Hlosses = AverageMeter()     # losses for hiding network
        Rlosses = AverageMeter()     # losses for reveal network
        Rlosses_ = AverageMeter()    # losses for reveal network with fake key
        SumLosses = AverageMeter()   # losses sumed by H and R with a factor beta(0.75 for default)
        Hdiff = AverageMeter()       # APD for hiding network (between container and cover)
        Rdiff = AverageMeter()       # APD for reveal network (between rev_secret and secret)
        Rdiff_ = AverageMeter()      # APD for reveal network (between rev_secret_ and zeors)
        # turn on training mode
        Hnet.train()
        Rnet.train()

        start_time = time.time()

        for i, (secret_image, cover_image) in enumerate(train_loader, start=1):
            data_time.update(time.time() - start_time)
            batch_size = config.batch_size

            cover_image, container_image, secret_image, rev_secret_image, rev_secret_image_, H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_ \
                = forward_pass(secret_image, cover_image, Hnet, Rnet, Anet, criterion, cover_dependent, key)
            
            Hlosses.update(H_loss.item(), batch_size)
            Rlosses.update(R_loss.item(), batch_size)
            Hdiff.update(H_diff.item(), batch_size)
            Rdiff.update(R_diff.item(), batch_size)
            
            if key is not None:
                Rlosses_.update(R_loss_.item(), batch_size)
                Rdiff_.update(R_diff_.item(), batch_size)

            loss_sum = H_loss + config.beta * R_loss + config.gamma * R_loss_
            SumLosses.update(loss_sum.item(), batch_size)

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            log = "[%02d/%d] [%04d/%d]\tH_loss: %.6f R_loss: %.6f R_loss_:%.6f H_diff: %.4f R_diff: %.4f R_diff_: %.4f\tdata_time: %.4f batch_time: %.4f" % (
                epoch, config.epochs, i, config.iters_per_epoch,
                Hlosses.val, Rlosses.val, Rlosses_.val,
                Hdiff.val, Rdiff.val, Rdiff_.val,
                data_time.val, batch_time.val
            )

            if i % config.log_freq == 0:
                print(log)
            if epoch == 0 and i % config.result_pic_freq == 0:
                save_result_pic(
                    batch_size,
                    cover_image, container_image,
                    secret_image, rev_secret_image, rev_secret_image_,
                    epoch, i,
                    config.train_pics_save_path
                )
            if i == config.iters_per_epoch:
                break

        save_result_pic(
            batch_size,
            cover_image, container_image,
            secret_image, rev_secret_image, rev_secret_image_,
            epoch, i,
            config.train_pics_save_path
        )
        epoch_log = "Training Epoch[%02d]\tSumloss=%.6f\tHloss=%.6f\tRloss=%.6f\tRloss_=%.6f\tHdiff=%.4f\tRdiff=%.4f\tRdiff_=%.4f\tlr= %.6f\tEpoch Time=%.4f" % (
            epoch, SumLosses.avg,
            Hlosses.avg, Rlosses.avg, Rlosses_.avg,
            Hdiff.avg, Rdiff.avg, Rdiff_.avg,
            optimizer.param_groups[0]['lr'],
            batch_time.sum
        )
        print_log(epoch_log)

        h_losses_list.append(Hlosses.avg)
        r_losses_list.append(Rlosses.avg)
        if key is not None:
            r_losses_list_.append(Rlosses_.avg)
        save_loss_pic(h_losses_list, r_losses_list, r_losses_list_, config.train_loss_save_path)

        val_hloss, val_rloss, val_rloss_, val_hdiff, val_rdiff, val_rdiff_ \
            = test(val_loader, Hnet, Rnet, Anet, criterion, cover_dependent, save_num=1, key=key, mode='val', epoch=epoch)

        scheduler.step(val_rloss)

        sum_diff = val_hdiff + val_rdiff
        is_best = sum_diff < MIN_LOSS
        MIN_LOSS = min(MIN_LOSS, sum_diff)

        if is_best:
            print_log("Save the best checkpoint: epoch%03d" % epoch)
            save_checkpoint(
                {
                    'epoch': epoch+1,
                    'H_state_dict': Hnet.state_dict(),
                    'R_state_dict': Rnet.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, is_best=True
            )
        else:
            print_log("Save the newest checkpoint: epoch%03d" % epoch)
            save_checkpoint(
                {
                    'epoch': epoch+1,
                    'H_state_dict': Hnet.state_dict(),
                    'R_state_dict': Rnet.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, is_best=False  # the newest
            )
    print("######## TRAIN END ########")


def test(data_loader, Hnet, Rnet, Anet, criterion, cover_dependent, save_num, key, mode, epoch=None):
    """Validate or test the performance of Hnet and Rnet.

    Parameters:
        data_loader (zip)      -- data_loader for secret and cover images
        Hnet (nn.Module)       -- hiding network
        Rnet (nn.Module)       -- reveal network
        Anet (nn.Module)       -- attack network, i.e. noise layers
        criterion              -- loss function to quantify the performation
        cover_dependent (bool) -- DDH (dependent deep hiding) or UDH (universal deep hiding)
        save_num (int)         -- the number of saved pictures
        key (torch.Tensor)     -- secure key (`None` denotes no key)
        mode (string)          -- validation or test mode [val | test] (val mode doesn't use fake key)
        epoch (int)            -- which epoch (for validation)
    """
    assert mode in ['test', 'val']

    print("\n#### %s begin ####" % mode)
    batch_size = config.batch_size
    # test information
    Hlosses = AverageMeter()     # losses for hiding network
    Rlosses = AverageMeter()     # losses for reveal network
    Rlosses_ = AverageMeter()    # losses for reveal network with fake key
    Hdiff = AverageMeter()       # APD for hiding network (between container and cover)
    Rdiff = AverageMeter()       # APD for reveal network (between rev_secret and secret)
    Rdiff_ = AverageMeter()      # APD for reveal network (between rev_secret_ and zeros)

    # turn on val mode
    Hnet.eval()
    Rnet.eval()

    for i, (secret_image, cover_image) in enumerate(data_loader, start=1):
        cover_image, container_image, secret_image, rev_secret_image, rev_secret_image_, H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_ \
            = forward_pass(secret_image, cover_image, Hnet, Rnet, Anet, criterion, cover_dependent, key)

        Hlosses.update(H_loss.item(), batch_size)
        Rlosses.update(R_loss.item(), batch_size)
        Hdiff.update(H_diff.item(), batch_size)
        Rdiff.update(R_diff.item(), batch_size)
        
        if key is not None:
            Rlosses_.update(R_loss_.item(), batch_size)
            Rdiff_.update(R_diff_.item(), batch_size)

        if i <= save_num:
            if mode == 'test':
                save_result_pic(
                    batch_size,
                    cover_image, container_image,
                    secret_image, rev_secret_image, rev_secret_=None,  # do not use randomly fake key in test mode
                    epoch=None, i=i,  # epoch is None in test mode
                    save_path=config.test_pics_save_path
                )
            else:
                save_result_pic(
                    batch_size,
                    cover_image, container_image,
                    secret_image, rev_secret_image, rev_secret_image_,
                    epoch, i,
                    config.val_pics_save_path
                )

    if mode == 'test':
        log = 'Test\tHloss=%.6f\tRloss=%.6f\tHdiff=%.4f\tRdiff=%.4f' % (
            Hlosses.avg, Rlosses.avg,
            Hdiff.avg, Rdiff.avg
        )
    else:
        log = 'Validation[%02d]\tHloss=%.6f\tRloss=%.6f\tRloss_=%.6f\tHdiff=%.4f\tRdiff=%.4f\tRdiff_=%.4f' % (
            epoch,
            Hlosses.avg, Rlosses.avg, Rlosses_.avg,
            Hdiff.avg, Rdiff.avg, Rdiff_.avg
        )
    print_log(log)
    print("#### %s end ####\n" % mode)
    return Hlosses.avg, Rlosses.avg, Rlosses_.avg, Hdiff.avg, Rdiff.avg, Rdiff_.avg
