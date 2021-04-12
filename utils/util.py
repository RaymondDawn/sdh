import os
import time
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.utils as vutils

from skimage.metrics import peak_signal_noise_ratio as _PSNR
from skimage.metrics import structural_similarity as _SSIM

from .options import opt, parser
from .image_folder import ImageFolder

MODIFIED_BITS = [128,96,80,64,48,32,16,8]  # length: 8

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


def md5(key: str) -> torch.Tensor:
    """Hash and binarize the key by MD5 algorithm."""
    hash_key = hashlib.md5(key.encode(encoding='UTF-8')).digest()
    binary_key = ''.join(format(x, '08b') for x in hash_key)
    tensor_key = torch.Tensor([float(x) for x in binary_key]).cuda()
    return tensor_key
STATIC_KEYS = [md5("hello world!")]


def weights_init(m):
    """Weights initialization for a network."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out') 
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


def print_log(log_info, log_path=opt.log_path, console=True):
    """Print log information to the console and log files."""
    if console:  # print the info into the console
        print(log_info)
    # write the log information into a log file
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
        fp.close()
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')


def print_network(net, save_path=opt.options_path):
    """Print network information."""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), save_path, console=False)
    print_log('Total number of parameters: %d\n' % num_params, save_path, console=False)


def save_options(save_path=opt.options_path):
    """Save options as a .txt file."""
    message = ''
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    print_log(message, save_path, console=False)


def save_checkpoint(state, is_best, save_path=opt.checkpoints_save_dir):
    """Save checkpoint files for training."""
    if is_best:  # best
        filename = '%s/checkpoint_best.pth.tar' % save_path
    else:  # newest
        filename = '%s/checkpoint_newest.pth.tar' % save_path
    torch.save(state, filename)


def save_image(input_image, image_path, save_all=False, start=0):
    """Save a 3D or 4D torch.Tensor as image(s) to the disk."""
    if not save_all:
        save_path, _ = os.path.split(image_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:  # save the whole batch images
        save_path = image_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    if isinstance(input_image, torch.Tensor):  # detach the tensor from current graph
        image_tensor = input_image.detach()
    else:
        raise TypeError("Type of the input should be `torch.Tensor`")
    
    if not save_all:
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        elif image_tensor.dim() != 3:
            raise TypeError('input_image should be 3D or 4D, but get a [%d]D tensor' % len(image_tensor.shape))
        
        image_numpy = image_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()  # add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)
    else:
        if image_tensor.dim() != 4:
            raise TypeError('input_image should be 4D if set `save_all` to True, but get a [%d]D tensor' % len(image_tensor.shape))
        for i in range(image_tensor.shape[0]):
            image_tensor_ = image_tensor[i]
            image_numpy = image_tensor_.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()  # add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            image_pil = Image.fromarray(image_numpy)
            image_pil.save(image_path + '/%d.png' % (i+start))


def save_result_pic(batch_size, cover, container, secret_set, rev_secret_set, rev_secret_, epoch, i, save_path):
    """Save a batch of result pictures."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if epoch is None:
        result_name = '%s/result_pic_batch%04d.png' % (save_path, i)
    else:
        result_name = '%s/result_pic_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    cover_gap = container - cover
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    show_cover = torch.cat((cover, container, cover_gap), dim=0)
    
    secret_gap_0 = rev_secret_set[0] - secret_set[0]
    secret_gap_set = [(secret_gap_0*10 + 0.5).clamp_(0.0, 1.0)]
    show_secret = torch.cat((secret_set[0], rev_secret_set[0], secret_gap_set[0]), dim=0)
    for i in range(1, opt.num_secrets):
        secret_gap_i = rev_secret_set[i] - secret_set[i]
        secret_gap_set.append((secret_gap_i*10 + 0.5).clamp_(0.0, 1.0))
        show_secret = torch.cat((show_secret, secret_set[i], rev_secret_set[i], secret_gap_set[i]), dim=0)
    if secret_gap_0.shape[1] == 1:  # gray
        show_secret = show_secret.repeat(1, 3, 1, 1)
    
    if rev_secret_ is None:
        show_all = torch.cat((show_cover, show_secret), dim=0)
    else:
        rev_secret_ = rev_secret_.repeat(1, 3//opt.channel_secret, 1, 1)
        show_all = torch.cat((show_cover, show_secret, (rev_secret_*50).clamp_(0.0, 1.0)), dim=0)

    # vutils.save_image(show_all, result_name, batch_size, padding=1, normalize=False)
    grid = vutils.make_grid(show_all, nrow=batch_size, padding=1, normalize=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(result_name)


def save_loss_pic(h_losses_list, r_losses_list, r_losses_list_, save_path=opt.loss_save_path):
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


def PSNR(batch_image0, batch_image1):
    """PSNR batch version for two tensors."""
    if isinstance(batch_image0, torch.Tensor) and isinstance(batch_image1, torch.Tensor):  # detach the tensor from current graph
        batch_image0_tensor = batch_image0.detach()
        batch_image1_tensor = batch_image1.detach()
    else:
        raise TypeError("Type of the input should be `torch.Tensor`")
    assert batch_image0_tensor.shape == batch_image1_tensor.shape
    
    SUM, b = 0, batch_image0_tensor.shape[0]
    for i in range(b):
        image0_numpy = batch_image0_tensor[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        image1_numpy = batch_image1_tensor[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        SUM += _PSNR(image0_numpy, image1_numpy)
    return SUM / b


def SSIM(batch_image0, batch_image1):
    """SSIM batch version for two tensors."""
    if isinstance(batch_image0, torch.Tensor) and isinstance(batch_image1, torch.Tensor):  # detach the tensor from current graph
        batch_image0_tensor = batch_image0.detach()
        batch_image1_tensor = batch_image1.detach()
    else:
        raise TypeError("Type of the input should be `torch.Tensor`")
    assert batch_image0_tensor.shape == batch_image1_tensor.shape
    
    SUM, b = 0, batch_image0_tensor.shape[0]
    for i in range(b):
        image0_numpy = batch_image0_tensor[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        image1_numpy = batch_image1_tensor[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        SUM += _SSIM(image0_numpy, image1_numpy, multichannel=True)
    return SUM / b


def adjust_learning_rate(optimizer, epoch, decay_num=2):
    """Set the learning rate to the initial LR decayed by `decay_num` every `lr_decay_freq` epochs."""
    lr = opt.lr * (1/decay_num ** (epoch // opt.lr_decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def forward_pass(secret, cover, Hnet, Rnet, Enet, NoiseLayers, criterion, cover_dependent, use_key=True, epoch=None, modified_bits=opt.modified_bits):
    """Forward propagation for hiding and reveal network and calculate losses and APD.
    
    Parameters:
        secret_image (torch.Tensor) -- secret images in batch
        cover_image (torch.Tensor)  -- cover images in batch
        Hnet (nn.Module)            -- hiding network
        Rnet (nn.Module)            -- reveal network
        criterion                   -- loss function
        cover_dependent (bool)      -- DDH (dependent deep hiding) or UDH (universal deep hiding)
        Enet (nn.Module or None)    -- a fully connected layer or `None`
    """
    cover, secret = cover.cuda(), secret.cuda()
    (b, c, h, w), (_, c_s, h_s, w_s) = cover.shape, secret.shape
    assert h == h_s and w == w_s
    
    secret_set, rev_secret_set = [], []
    for i in range(opt.num_secrets):
        secret_set.append(secret[b*i:b*(i+1), :, :, :])

    count_diff = 0
    # pre-processing for key(s)
    if use_key:
        key_set, red_key_set = [], []
        if opt.static_key:
            assert len(STATIC_KEYS) == opt.num_secrets
            key_set = STATIC_KEYS
        else:
            for i in range(opt.num_secrets):
                key_set.append(torch.Tensor([float(torch.randn(1)<0) for _ in range(w)]).cuda())
        
        if not opt.stage_modification:
            if opt.num_secrets == 1:
                fake_key, s = key_set[0].clone(), set()
                bits = np.random.randint(8,129)  # random stragety
                for j in range(bits):
                    index = (j + int(np.random.rand() * 128)) % 128
                    while index in s:
                        index = (j + int(np.random.rand() * 128)) % 128
                    s.add(index)
                    fake_key[index] = -fake_key[index] + 1  # 0->1; 1->0
            else:  # random generated
                fake_key = torch.Tensor([float(torch.randn(1)<0) for _ in range(w)]).cuda()
                # assert fake_key is different from every key
                for key in key_set:
                    while torch.equal(fake_key, key):
                        fake_key = torch.Tensor([float(torch.randn(1)<0) for _ in range(w)]).cuda()
        else: # stair strategy
            assert (opt.num_secrets == 1) and (epoch is not None)
            fake_key, s = key_set[0].clone(), set()
            bits = MODIFIED_BITS[epoch//10]
            for j in range(bits):
                index = (j + int(np.random.rand() * 128)) % 128
                while index in s:
                    index = (j + int(np.random.rand() * 128)) % 128
                s.add(index)
                fake_key[index] = -fake_key[index] + 1  # 0->1; 1->0

        if opt.num_secrets == 1:
            for i in range(128):
                count_diff += int(key_set[0][i] != fake_key[i])

        # redundance for key(s)
        for key in key_set:
            red_key_set.append(Enet(key))
        red_fake_key = Enet(fake_key)
        zeros = torch.zeros(red_key_set[0].shape).cuda()

    if cover_dependent:
        if use_key:
            H_input = cover
            for i in range(opt.num_secrets):
                H_input = torch.cat((H_input, secret_set[i], red_key_set[i]), dim=1)
        else:
            H_input = cover
            for i in range(opt.num_secrets):
                H_input = torch.cat((H_input, secret_set[i]), dim=1)
    else:
        if use_key:
            H_input = torch.cat((secret_set[0], red_key_set[0]), dim=1)
            for i in range(1, opt.num_secrets):
                H_input = torch.cat((H_input, secret_set[i], red_key_set[i]), dim=1)
        else:
            H_input = secret_set[0]
            for i in range(1, opt.num_secrets):
                H_input = torch.cat((H_input, secret_set[i]), dim=1)
    H_output = Hnet(H_input)

    if cover_dependent:
        container = H_output
    else:
        container = H_output + cover
    H_loss = criterion(container, cover)

    container = NoiseLayers(container)

    R_loss = 0
    if use_key:
        for i in range(opt.num_secrets):
            # modify key to test the sensitivity
            bits = modified_bits
            key_, s = key_set[i].clone(), set()
            for j in range(bits):
                index = (j + int(np.random.rand() * 128)) % 128
                while index in s:
                    index = (j + int(np.random.rand() * 128)) % 128
                s.add(index)
                key_[index] = -key_[index] + 1  # 0->1; 1->0
            red_key_ = Enet(key_)
            rev_secret_set.append(Rnet(torch.cat((container, red_key_), dim=1)))
            R_loss += criterion(rev_secret_set[i], secret_set[i])
    else:
        R_output = Rnet(container)
        for i in range(opt.num_secrets):
            rev_secret_set.append(R_output[:, c_s*i:c_s*(i+1), :, :])
            R_loss += criterion(secret_set[i], rev_secret_set[i])
    R_loss /= opt.num_secrets

    if use_key:
        rev_secret_ = Rnet(torch.cat((container, red_fake_key), dim=1))
        R_loss_ = criterion(rev_secret_, zeros)
        R_diff_ = (rev_secret_).abs().mean() * 255
    else:
        rev_secret_, R_loss_, R_diff_ = None, 0, 0

    # L1 metric (APD)
    H_diff = (container - cover).abs().mean() * 255
    R_diff = 0
    for i in range(opt.num_secrets):
        R_diff += (rev_secret_set[i] - secret_set[i]).abs().mean() * 255
    R_diff /= opt.num_secrets

    return cover, container, secret_set, rev_secret_set, rev_secret_, H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff


def train(train_loader_secret, train_loader_cover, val_loader_secret, val_loader_cover, Hnet, Rnet, Enet, NoiseLayers, optimizer, scheduler, criterion, cover_dependent, use_key, Adversary=None, optimizer_adv=None):
    """Train Hnet and Rnet and schedule learning rate by the validation results.
    
    Parameters:
        train_loader_secret     -- train_loader for secret images
        train_loader_cover      -- train_loader for cover images
        val_loader_secret       -- val_loader for secret images
        val_loader_cover        -- val_loader for cover images
        Hnet (nn.Module)        -- hiding network
        Rnet (nn.Module)        -- reveal network
        optimizer               -- optimizer for Hnet and Rnet
        scheduler               -- scheduler for optimizer to set dynamic learning rate
        criterion               -- loss function
        cover_dependent (bool)  -- DDH (dependent deep hiding) or UDH (universal deep hiding)
    """
    #### training and update parameters ####
    MIN_LOSS = float('inf')
    h_losses_list, r_losses_list, r_losses_list_ = [], [], []
    print("######## TRAIN BEGIN ########")
    for epoch in range(opt.start_epoch, opt.epochs):
        if opt.start_epoch != 0:
            assert opt.load_checkpoint
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
        Alosses = AverageMeter()     # losses for adversarial loss
        Alosses_ = AverageMeter()    # losses for adversarial network
        SumLosses = AverageMeter()   # losses sumed by H and R with a factor beta(0.75 for default)
        Hdiff = AverageMeter()       # APD for hiding network (between container and cover)
        Rdiff = AverageMeter()       # APD for reveal network (between rev_secret and secret)
        Rdiff_ = AverageMeter()      # APD for reveal network (between rev_secret_ and zeors)
        Count = AverageMeter()
        # turn on training mode
        Hnet.train()
        Rnet.train()

        start_time = time.time()

        for i, (secret, cover) in enumerate(train_loader, start=1):
            data_time.update(time.time() - start_time)
            batch_size = opt.batch_size

            cover, container, secret_set, rev_secret_set, rev_secret_, H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff \
                = forward_pass(secret, cover, Hnet, Rnet, Enet, NoiseLayers, criterion, cover_dependent, use_key, epoch)
            
            if Adversary is not None:
                criterion_adv = nn.MSELoss().cuda()
                TRUE, FALSE = torch.ones(batch_size, 1).cuda(), torch.zeros(batch_size, 1).cuda()
                loss_adv = criterion_adv(Adversary(cover), TRUE) + criterion_adv(Adversary(container.detach()), FALSE)
                Alosses_.update(loss_adv.item(), batch_size)

                optimizer_adv.zero_grad()
                loss_adv.backward()
                optimizer_adv.step()

            Hlosses.update(H_loss.item(), batch_size)
            Rlosses.update(R_loss.item(), batch_size)
            Hdiff.update(H_diff.item(), batch_size)
            Rdiff.update(R_diff.item(), batch_size)
            Count.update(count_diff, 1)
            if use_key:
                Rlosses_.update(R_loss_.item(), batch_size)
                Rdiff_.update(R_diff_.item(), batch_size)
            if Adversary is not None:
                A_loss = criterion_adv(Adversary(container), TRUE)
                Alosses.update(A_loss.item(), batch_size)
            else:
                A_loss = 0

            rate = (i+epoch*opt.iters_per_epoch) / (50*opt.iters_per_epoch) if epoch < 50 else 1
            loss_sum = H_loss + opt.beta * R_loss + opt.gamma * R_loss_ + opt.delta * rate * A_loss

            SumLosses.update(loss_sum.item(), batch_size)

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            log = "[%02d/%d] [%04d/%d]\tH_loss: %.6f R_loss: %.6f R_loss_:%.6f H_diff: %.4f R_diff: %.4f R_diff_: %.4f\tA_loss: %.6f A_loss_: %.6f\tCount: %.4f\tdata_time: %.4f batch_time: %.4f" % (
                epoch, opt.epochs, i, opt.iters_per_epoch,
                Hlosses.val, Rlosses.val, Rlosses_.val,
                Hdiff.val, Rdiff.val, Rdiff_.val,
                Alosses.val, Alosses_.val, Count.val,
                data_time.val, batch_time.val
            )

            if i % opt.log_freq == 0:
                print(log)
            if epoch == 0 and i % opt.result_pic_freq == 0:
                save_result_pic(
                    batch_size,
                    cover, container,
                    secret_set, rev_secret_set, rev_secret_,
                    epoch, i,
                    opt.train_pics_save_dir
                )
            if i == opt.iters_per_epoch:
                break

        save_result_pic(
            batch_size,
            cover, container,
            secret_set, rev_secret_set, rev_secret_,
            epoch, i,
            opt.train_pics_save_dir
        )
        epoch_log = "Training Epoch[%02d]\tSumloss=%.6f\tHloss=%.6f\tRloss=%.6f\tRloss_=%.6f\tHdiff=%.4f\tRdiff=%.4f\tRdiff_=%.4f\tALoss=%.6f\tALoss_=%.6f\tCount=%.4f\tlr= %.6f\tEpoch Time=%.4f" % (
            epoch, SumLosses.avg,
            Hlosses.avg, Rlosses.avg, Rlosses_.avg,
            Hdiff.avg, Rdiff.avg, Rdiff_.avg,
            Alosses.avg, Alosses_.avg, Count.avg,
            optimizer.param_groups[0]['lr'],
            batch_time.sum
        )
        print_log(epoch_log)

        h_losses_list.append(Hlosses.avg)
        r_losses_list.append(Rlosses.avg)
        if use_key:
            r_losses_list_.append(Rlosses_.avg)
        save_loss_pic(h_losses_list, r_losses_list, r_losses_list_, opt.loss_save_path)

        val_hloss, val_rloss, val_rloss_, val_hdiff, val_rdiff, val_rdiff_ \
            = inference(val_loader, Hnet, Rnet, Enet, NoiseLayers, criterion, cover_dependent, use_key, save_num=1, mode='val', epoch=epoch)

        scheduler.step(val_rloss)

        sum_diff = val_hdiff + val_rdiff + val_rdiff_
        is_best = sum_diff < MIN_LOSS
        MIN_LOSS = min(MIN_LOSS, sum_diff)

        state = 'best' if is_best else 'newest'
        print_log("Save the %s checkpoint: epoch%03d" % (state, epoch))
        if Adversary is not None:
            save_checkpoint(
                {
                    'epoch': epoch+1,
                    'H_state_dict': Hnet.state_dict(),
                    'R_state_dict': Rnet.state_dict(),
                    'E_state_dict': Enet.state_dict(),
                    'Adversary_state_dict': Adversary.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, is_best=is_best
            )
        else:
            save_checkpoint(
                {
                    'epoch': epoch+1,
                    'H_state_dict': Hnet.state_dict(),
                    'R_state_dict': Rnet.state_dict(),
                    'E_state_dict': Enet.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, is_best=is_best
            )
    print("######## TRAIN END ########")


def inference(data_loader, Hnet, Rnet, Enet, NoiseLayers, criterion, cover_dependent, use_key, save_num=1, mode='test', epoch=None):
    """Validate or test the performance of Hnet and Rnet.

    Parameters:
        data_loader (zip)      -- data_loader for secret and cover images
        Hnet (nn.Module)       -- hiding network
        Rnet (nn.Module)       -- reveal network
        criterion              -- loss function to quantify the performation
        cover_dependent (bool) -- DDH (dependent deep hiding) or UDH (universal deep hiding)
        save_num (int)         -- the number of saved pictures
        mode (string)          -- validation or test mode [val | test] (val mode doesn't use fake key)
        epoch (int)            -- which epoch (for validation)
    """
    assert mode in ['val', 'test'], '`mode` is expected to be `val` or `test`'

    print("\n#### %s begin ####" % mode)
    batch_size = opt.batch_size
    # test information
    Hlosses = AverageMeter()     # losses for hiding network
    Rlosses = AverageMeter()     # losses for reveal network
    Rlosses_ = AverageMeter()    # losses for reveal network with fake key
    Hdiff = AverageMeter()       # APD for hiding network (between container and cover)
    Rdiff = AverageMeter()       # APD for reveal network (between rev_secret and secret)
    Rdiff_ = AverageMeter()      # APD for reveal network (between rev_secret_ and zeros)
    Count = AverageMeter()

    # turn on val mode
    Hnet.eval()
    Rnet.eval()

    for i, (secret, cover) in enumerate(data_loader, start=1):
        cover, container, secret_set, rev_secret_set, rev_secret_, H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff \
            = forward_pass(secret, cover, Hnet, Rnet, Enet, NoiseLayers, criterion, cover_dependent, use_key, epoch)

        Hlosses.update(H_loss.item(), batch_size)
        Rlosses.update(R_loss.item(), batch_size)
        Hdiff.update(H_diff.item(), batch_size)
        Rdiff.update(R_diff.item(), batch_size)
        Count.update(count_diff, 1)
        if use_key:
            Rlosses_.update(R_loss_.item(), batch_size)
            Rdiff_.update(R_diff_.item(), batch_size)

        if i <= save_num:
            if mode == 'test':
                save_result_pic(
                    batch_size,
                    cover, container,
                    secret_set, rev_secret_set, rev_secret_,
                    epoch=None, i=i,  # epoch is None in test mode
                    save_path=opt.test_pics_save_dir
                )
            else:
                save_result_pic(
                    batch_size,
                    cover, container,
                    secret_set, rev_secret_set, rev_secret_,
                    epoch, i,
                    opt.val_pics_save_dir
                )

    if mode == 'test':
        log = 'Test\tHloss=%.6f\tRloss=%.6f\tRloss_=%.6f\tHdiff=%.4f\tRdiff=%.4f\tRdiff_=%.4f\tCount=%.4f' % (
            Hlosses.avg, Rlosses.avg, Rlosses_.avg,
            Hdiff.avg, Rdiff.avg, Rdiff_.avg, Count.avg
        )
    else:
        log = 'Validation[%02d]\tHloss=%.6f\tRloss=%.6f\tRloss_=%.6f\tHdiff=%.4f\tRdiff=%.4f\tRdiff_=%.4f\tCount=%.4f' % (
            epoch,
            Hlosses.avg, Rlosses.avg, Rlosses_.avg,
            Hdiff.avg, Rdiff.avg, Rdiff_.avg, Count.avg
        )
    print_log(log)
    print("#### %s end ####\n" % mode)
    return Hlosses.avg, Rlosses.avg, Rlosses_.avg, Hdiff.avg, Rdiff.avg, Rdiff_.avg
