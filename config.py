import time
import torch


ngpu                  = torch.cuda.device_count()
workers               = ngpu * 4
image_size            = 128
training_dataset_size = 25000

analysis              = False
test                  = False

exper_name            = 'experiment0'  # time.strftime('%Y-%m-%d_%H-%M', time.localtime())
checkpoint_mode       = 'none'  # ['none' | 'best' | 'newest']
ROOT                  = '/content/drive/MyDrive'
DATA_DIR              = ROOT + '/dataset'
experiment_dir        = ROOT + '/sdh/exper_info/' + exper_name
if not analysis:
    config_path       = experiment_dir + '/config.txt'
    log_path          = experiment_dir + '/log.txt'
else:
    config_path       = experiment_dir + '/analysis/%s/config.txt' % checkpoint_mode
    log_path          = experiment_dir + '/analysis/%s/log.txt' % checkpoint_mode
checkpoint_save_path  = experiment_dir + '/checkpoint'
train_pics_save_path  = experiment_dir + '/train_pics'
train_loss_save_path  = experiment_dir + '/train_loss.png'
val_pics_save_path    = experiment_dir + '/val_pics'
test_pics_save_path   = experiment_dir + '/test_pics'
anal_pics_save_path   = experiment_dir + '/analysis/%s/pics' % checkpoint_mode
if checkpoint_mode == 'none':
    checkpoint        = ''
else:
    checkpoint        = '/checkpoint_%s.pth.tar' % checkpoint_mode
checkpoint_path       = checkpoint_save_path + checkpoint

epochs                = 50
batch_size            = 25
beta                  = 0.75
gamma                 = 0.25
lr                    = 0.001
lr_decay_freq         = 30
iters_per_epoch       = training_dataset_size // batch_size

log_freq              = 10
result_pic_freq       = 100

noise                 = False
noise_type            = 'combine'  # ['combine' | 'noise' | 'blur' | 'resize' | 'jpeg']
key                   = 'Hell0_World'
hash_algorithm        = 'md5'  # ['md5' | 'sha256' | 'sha512']
key_redundance_size   = image_size // 8
cover_dependent       = False
channel_secret        = 3
channel_cover         = 3
num_downs             = 5
norm_type             = 'batch'  # ['batch' | 'instance' | 'none']
loss                  = 'l2'  # ['l2' | 'l1']
