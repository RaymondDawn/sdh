import os
import time
import shutil
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


def print_network(net):
    """Print network information."""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


def save_config():
    """Save configuations as .txt file."""
    fp = open(config.config_path, "w")
    fp.writelines("ngpu\t\t\t\t%d\n" % config.ngpu)
    fp.writelines("workers\t\t\t\t%d\n" % config.workers)
    fp.writelines("image_size\t\t\t\t%d\n" % config.image_size)
    fp.writelines("training_dataset_size\t\t\t\t%d\n" % config.training_dataset_size)
    fp.writelines("cur_time\t\t\t\t%s\n" % config.cur_time)
    fp.writelines("DATA_DIR\t\t\t\t%s\n" % config.DATA_DIR)
    fp.writelines("experiment_dir\t\t\t\t%s\n" % config.experiment_dir)
    fp.writelines("config_path\t\t\t\t%s\n" % config.config_path)
    fp.writelines("log_path\t\t\t\t%s\n" % config.log_path)
    fp.writelines("checkpoint_path\t\t\t\t%s\n" % config.checkpoint_path)
    fp.writelines("train_pics_save_path\t\t\t\t%s\n" % config.train_pics_save_path)
    fp.writelines("val_pics_save_path\t\t\t\t%s\n" % config.val_pics_save_path)
    fp.writelines("test_path\t\t\t\t%s\n" % config.test_path)
    fp.writelines("test_pics_save_path\t\t\t\t%s\n" % config.test_pics_save_path)
    fp.writelines("checkpoint\t\t\t\t%s\n" % config.checkpoint)
    fp.writelines("checkpoint_diff\t\t\t\t%s\n" % config.checkpoint_diff)
    fp.writelines("epochs\t\t\t\t%d\n" % config.epochs)
    fp.writelines("batch_size\t\t\t\t%d\n" % config.batch_size)
    fp.writelines("beta\t\t\t\t%d\n" % config.beta)
    fp.writelines("lr\t\t\t\t%d\n" % config.lr)
    fp.writelines("lr_decay_freq\t\t\t\t%d\n" % config.lr_decay_freq)
    fp.writelines("iters_per_epoch\t\t\t\t%d\n" % config.iters_per_epoch)
    fp.writelines("log_freq\t\t\t\t%d\n" % config.log_freq)
    fp.writelines("result_pic_freq\t\t\t\t%d\n" % config.result_pic_freq)
    fp.writelines("num_downs\t\t\t\t%d\n" % config.num_downs)
    fp.writelines("cover_dependent\t\t\t\t%s\n" % config.cover_dependent)
    fp.writelines("channel_secret\t\t\t\t%d\n" % config.channel_secret)
    fp.writelines("channel_cover\t\t\t\t%d\n" % config.channel_cover)
    fp.writelines("num_downs\t\t\t\t%d\n" % config.num_downs)
    fp.writelines("norm_type\t\t\t\t%s\n" % config.norm_type)
    fp.writelines("loss\t\t\t\t%s\n" % config.loss)
    fp.close()


def save_checkpoint(state, is_best, epoch, prefix):
    """Save checkpoint files for training"""
    filename = '%s/checkpoints_%03d.pth.tar' % (config.checkpoint_path, epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/best_checkpoint_%03d.pth.tar' % (config.checkpoint_path, epoch))


def save_result_pic(batch_size, cover, container, secret, rev_secret, epoch, i, save_path):
    result_name = '%s/result_pic_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    cover_gap = container - cover
    secret_gap = rev_secret - secret
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    show_cover = torch.cat((cover, container, cover_gap), dim=0)
    show_secret = torch.cat((secret, rev_secret, secret_gap), dim=0)
    show_all = torch.cat((show_cover, show_secret), dim=0)

    vutils.save_image(show_all, result_name, batch_size, padding=1, normalize=True)


def adjust_learning_rate(optimizer, epoch):
    """Set the learning rate to the initial LR decayed by 10 every `lr_decay_freq` epochs."""
    lr = config.lr * (0.1 ** (epoch // config.lr_decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def forward_pass(secret_image, cover_image, Hnet, Rnet, criterion, cover_dependent=False):
    """Forward propagation for hiding and reveal network and calculate losses and APD.
    
    Parameters:
        secret_image (torch.Tensor) -- secret images in batch
        cover_image (torch.Tensor)  -- cover images in batch
        Hnet (nn.Module)            -- hiding network
        Rnet (nn.Module)            -- reveal network
        criterion                   -- loss function
        cover_dependent (bool)      -- DDH (dependent deep hiding) or UDH (universal deep hiding)
    """
    cover_image = cover_image.cuda()
    secret_image = secret_image.cuda()

    if cover_dependent:
        H_input = torch.cat((cover_image, secret_image), dim=1)
    else:
        H_input = secret_image

    H_output = Hnet(H_input)

    if cover_dependent:
        container_image = H_output
    else:
        container_image = H_output + cover_image

    H_loss = criterion(container_image, cover_image)

    rev_secret_image = Rnet(container_image)
    R_loss = criterion(rev_secret_image, secret_image)

    # L1 metric (APD: average pixel difference)
    H_diff = (container_image - cover_image).abs().mean() * 255
    R_diff = (rev_secret_image - secret_image).abs().mean() * 255

    return cover_image, container_image, secret_image, rev_secret_image, H_loss, R_loss, H_diff, R_diff
    

def validation(val_loader, epoch, Hnet, Rnet, criterion):
    print_log("#### validation begin ####")
    batch_size = config.batch_size
    # validation information
    batch_time = AverageMeter()  # time for processing a batch
    Hlosses = AverageMeter()     # losses for hiding network
    Rlosses = AverageMeter()     # losses for reveal network
    SumLosses = AverageMeter()   # losses sumed by H and R with a factor beta(0.75 for default)
    Hdiff = AverageMeter()       # APD for hiding network (between container and cover)
    Rdiff = AverageMeter()       # APD for reveal network (between rev_secret and secret)
    # turn on val mode
    Hnet.eval()
    Rnet.eval()

    start_time = time.time()
    
    for i, (secret_image, cover_image) in enumerate(val_loader, start=1):
        cover_image, container_image, secret_image, rev_secret_image, H_loss, R_loss, H_diff, R_diff \
                = forward_pass(secret_image, cover_image, Hnet, Rnet, criterion, cover_dependent)
            
        Hlosses.update(H_loss.item(), batch_size)
        Rlosses.update(R_loss.item(), batch_size)
        Hdiff.update(H_diff.item(), batch_size)
        Rdiff.update(R_diff.item(), batch_size)

        if i == 1:
            save_result_pic(
                batch_size,
                cover_image, container_image,
                secret_image, rev_secret_image,
                epoch, i,
                config.val_pics_save_path
            )
        
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        val_log = 'Validation[%d] val_Hloss: %.6f val_Rloss: %.6f val_Hdiff:%.4f val_Rdiff: %.4f\tbatch time: %.2f' % (
            epoch,
            Hlosses.val, Rlosses.val,
            Hdiff.val, Rdiff.val,
            batch_time.val
        )
        if i % config.log_freq == 1:
            print(val_log)

    val_log = 'Validation[%d] val_Hloss: %.6f val_Rloss: %.6f val_Hdiff:%.4f val_Rdiff: %.4f\tbatch time: %.2f' % (
        epoch,
        Hlosses.avg, Rlosses.avg,
        Hdiff.avg, Rdiff.avg,
        batch_time.sum
    )
    print_log(val_log)

    print_log("#### validation end ####")
    return Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg


def train(train_loader, val_loader, Hnet, Rnet, optimizer, scheduler, criterion, cover_dependent=False):
    """Train Hnet and Rnet and schedule learning rate by the validation results.
    
    Parameters:
        train_loader (zip)     -- zip object combined by secret and cover train_loader
        val_loader (zip)       -- zip object combined by secret and cover val_loader
        Hnet (nn.Module)       -- hiding network
        Rnet (nn.Module)       -- reveal network
        optimizer              -- optimizer for Hnet and Rnet
        scheduler              -- scheduler for optimizer to set dynamic learning rate
        criterion              -- loss function
        cover_dependent (bool) -- DDH (dependent deep hiding) or UDH (universal deep hiding)
    """
    # training information
    batch_time = AverageMeter()  # time for processing a batch
    data_time = AverageMeter()   # time for reading data
    Hlosses = AverageMeter()     # losses for hiding network
    Rlosses = AverageMeter()     # losses for reveal network
    SumLosses = AverageMeter()   # losses sumed by H and R with a factor beta(0.75 for default)
    Hdiff = AverageMeter()       # APD for hiding network (between container and cover)
    Rdiff = AverageMeter()       # APD for reveal network (between rev_secret and secret)
    # turn on training mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()

    #### training and update parameters ####
    MIN_LOSS = 0x3f3f3f3f
    print_log("######## TRAIN BEGIN ########")
    for epoch in range(config.epochs):
        adjust_learning_rate(optimizer, epoch)

        for i, (secret_image, cover_image) in enumerate(train_loader, start=1):
            data_time.update(time.time() - start_time)
            batch_size = config.batch_size

            cover_image, container_image, secret_image, rev_secret_image, H_loss, R_loss, H_diff, R_diff \
                = forward_pass(secret_image, cover_image, Hnet, Rnet, criterion, cover_dependent)
            
            Hlosses.update(H_loss.item(), batch_size)
            Rlosses.update(R_loss.item(), batch_size)
            Hdiff.update(H_diff.item(), batch_size)
            Rdiff.update(R_diff.item(), batch_size)

            loss_sum = H_loss + config.beta * R_loss
            SumLosses.update(loss_sum.item(), batch_size)

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            batch_time.update(time.time() - start_time)

            start_time = time.time()
            log = '[%d/%d] [%d/%d]\tH_loss: %.6f R_loss: %.6f H_diff: %.4f R_diff: %.4f\tdata_time: %.4f\tbatch_time: %.4f' % (
                epoch, config.epochs, i, config.iters_per_epoch,
                Hlosses.val, Rlosses.val, Hdiff.val, Rdiff.val,
                data_time.val, batch_time.val
            )

            if i % config.log_freq == 0:
                print(log)
            if epoch == 0 and i % config.result_pic_freq == 0:
                save_result_pic(
                    batch_size,
                    cover_image.detach(), container_image.detach(),
                    secret_image.detach(), rev_secret_image.detach(),
                    epoch, i,
                    config.train_pics_save_path
                )
            if i == config.iters_per_epoch:
                break

        save_result_pic(
            batch_size,
            cover_image.detach(), container_image.detach(),
            secret_image.detach(), rev_secret_image.detach(),
            epoch, i,
            config.train_pics_save_path
        )
        epoch_log = "\nTraining Epoch[%d]\tHloss=%.6f\tRloss=%.6f\tHdiff=%.4f\tRdiff=%.4f\tlr= %.6f\tEpoch Time=%.4f" % (
            epoch,
            Hlosses.avg, Rlosses.avg,
            Hdiff.avg, Rdiff.avg,
            optimizer.param_groups[0]['lr'],
            batch_time.sum
        )
        print_log(epoch_log)

        #### validation, schedule learning rate and make checkpoint ####
        val_hloss, val_rloss, val_hdiff, val_rdiff = validation(val_loader, epoch, Hnet, Rnet, criterion)

        scheduler.step(val_rloss)

        sum_diff = val_hdiff + val_rdiff
        is_best = sum_diff < MIN_LOSS
        MIN_LOSS = min(MIN_LOSS, sum_diff)

        save_checkpoint(
            {
                'epoch': epoch+1,
                'H_state_dict': Hnet.state_dict(),
                'R_state_dict': Rnet.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            is_best, epoch,
            '%s/epoch_%d_Hloss_%.4f_Rloss%.4f_Hdiff%.4f_Rdiff%.4f' % (
                config.checkpoint_path, epoch,
                val_hloss, val_rloss,
                val_hdiff, val_rdiff
            )
        )
    print_log("######## TRAIN END ########")
