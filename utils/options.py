import os
import time
import argparse
import torch

parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0; 0,1,2; 0,2. use -1 for CPU')
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--exper_name', type=str, default=str(time.strftime('%Y-%m-%d_%H-%M', time.localtime())), help='experiment name')
parser.add_argument('--root', type=str, default='/content/drive/MyDrive', help='root dir of this project')
parser.add_argument('--exper_dir', type=str, default='', help='dir of one experiment')
parser.add_argument('--options_path', type=str, default='', help='path of options')
parser.add_argument('--log_path', type=str, default='', help='path of log information')
parser.add_argument('--checkpoints_save_dir', type=str, default='', help='dir of saving checkopints')
parser.add_argument('--train_pics_save_dir', type=str, default='', help='dir of saving pictures in training')
parser.add_argument('--val_pics_save_dir', type=str, default='', help='dir of saving pictures in validation')
parser.add_argument('--test_pics_save_dir', type=str, default='', help='dir of saving pictures in test')
parser.add_argument('--analysis_pics_save_dir', type=str, default='', help='dir of saving pictures in analysis')
parser.add_argument('--loss_save_path', type=str, default='', help='path of saving the curve of loss function')
parser.add_argument('--log_freq', type=int, default=10, help='frequency of saving log information')
parser.add_argument('--result_pic_freq', type=int, default=100, help='frequency of saving result pictures in the first epoch')

# dataset parameters
parser.add_argument('--image_size', type=int, default=128, help='size of images')
parser.add_argument('--training_dataset_size', type=int, default=25000, help='size of training dataset')
parser.add_argument('--data_dir', type=str, default='', help='dir of dataset')

# model parameters
parser.add_argument('--cover_dependent', action='store_true', help='DDH(True) or UDH(False)')
parser.add_argument('--use_key', action='store_true', help='use key or not')
parser.add_argument('--channel_cover', type=int, default=3, help='number of channels for cover images')
parser.add_argument('--channel_secret', type=int, default=3, help='number of channels for secret images')
parser.add_argument('--channel_key', type=int, default=3, help='number of channels for embedded key')
parser.add_argument('--num_downs', type=int, default=5, help='number of down submodules in U-Net')
parser.add_argument('--norm_type', type=str, default='batch', help='type of normalization layer')
parser.add_argument('--loss', type=str, default='l2', help='loss function [l1 | l2]')
parser.add_argument('--num_secrets', type=int, default=1, help='the number of secret images to be hidden')

# training parameters
parser.add_argument('--epochs', type=int, default=80, help='epochs for training')
parser.add_argument('--start_epoch', type=int, default=0, help='the start epoch to continue training')
parser.add_argument('--batch_size', type=int, default=25, help='batch size')
parser.add_argument('--beta', type=float, default=0.75, help='weight of secret reveal')
parser.add_argument('--gamma', type=float, default=0.5, help='weight of fake_key reveal')
parser.add_argument('--delta', type=float, default=0.001, help='weight of adversarial classfication')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay_freq', type=int, default=30, help='frequency of decaying lr')
parser.add_argument('--iters_per_epoch', type=int, default=1000, help='number of iterations in one epoch')
parser.add_argument('--noise_type', type=str, default='identity', help='type of distortion [identity | noise | blur | resize | jpeg | combine]')
parser.add_argument('--adversary', action='store_true', help='use adversarial structure')
parser.add_argument('--redundance', type=int, default=32, help='redundance size of key; e.g. `16` for mapping it to a 3*16*16 tensor; `-1` for simple duplication')
parser.add_argument('--generation_type', default='stair', help='a type of fake key [stair | random | custom | ELSE]')
parser.add_argument('--static_key', action='store_true', help='use s static key')
parser.add_argument('--explicit', action='store_true', help='extract secret explicitly')

# additional parameters
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--load_checkpoint', action='store_true', help='load checkpoint')
parser.add_argument('--checkpoint_name', type=str, default='', help='exper_name of loaded checkpoint')
parser.add_argument('--checkpoint_type', type=str, default='best', help='type of the checkpint file [best | newest]')
parser.add_argument('--checkpoint_path', type=str, default='', help='path of one checkpint file')
parser.add_argument('--key', type=str, default='hello world!', help='genuine key')
parser.add_argument('--fake_key', type=str, default='this is a fake key', help='fake key')
parser.add_argument('--modified_bits', type=int, default=0, help='number of modified bits in the key')
parser.add_argument('--feature_map', action='store_true', help='show the feature maps of Rnet')


opt = parser.parse_args()

_ngpu = len(opt.gpu_ids.split(','))
assert _ngpu <= torch.cuda.device_count(), 'There are not enough GPUs!'

opt.iters_per_epoch = opt.training_dataset_size // opt.batch_size

_r = opt.redundance
assert (_r == -1) or (_r % 2 == 0 and _r >= 8), 'Unexpected redundance size!'

if opt.checkpoint_name == '':
    opt.checkpoint_name = opt.exper_name
_load_checkpoint_dir = opt.root +  '/sdh/exper_info/' + opt.checkpoint_name
assert _load_checkpoint_dir, 'Do not exist this checkpoint dir!'

if opt.test:
    opt.load_checkpoint = True
    assert opt.load_checkpoint, 'Test mode must load a checkpoint file'

opt.data_dir = opt.root + '/dataset'
opt.exper_dir = opt.root +  '/sdh/exper_info/' + opt.exper_name
opt.options_path = opt.exper_dir +  '/options.txt'
opt.log_path = opt.exper_dir +  '/log.txt'
opt.checkpoints_save_dir = opt.exper_dir + '/checkpoints'
opt.train_pics_save_dir = opt.exper_dir + '/train_pics'
opt.val_pics_save_dir = opt.exper_dir + '/val_pics'
opt.test_pics_save_dir = opt.exper_dir + '/test_pics'
opt.analysis_pics_save_dir = opt.exper_dir + '/analysis_pics'
opt.loss_save_path = opt.exper_dir + '/train_loss.png'
opt.checkpoint_path = _load_checkpoint_dir + '/checkpoints/checkpoint_%s.pth.tar' % opt.checkpoint_type
