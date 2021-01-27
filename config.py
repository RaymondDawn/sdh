import time
import torch


ngpu                  = torch.cuda.device_count()
workers               = ngpu * 4
image_size            = 128
training_dataset_size = 25000

exper_name            = '2021-01-26_08-37'#time.strftime('%Y-%m-%d_%H-%M', time.localtime())
ROOT                  = '/content/drive/MyDrive'
DATA_DIR              = ROOT + '/dataset'
experiment_dir        = ROOT + '/sdh/exper_info/' + exper_name
config_path           = experiment_dir + "/config.txt"
log_path              = experiment_dir + '/train_log.txt'
checkpoint_save_path  = experiment_dir + '/checkpoint'
train_pics_save_path  = experiment_dir + '/train_pics'
train_loss_save_path  = experiment_dir + '/train_loss.png'
val_pics_save_path    = experiment_dir + '/val_pics'
test_pics_save_path   = experiment_dir + '/test_pics'
checkpoint            = '/checkpoint_newest.pth.tar'
checkpoint_path       = checkpoint_save_path + checkpoint
test                  = True

epochs                = 50
batch_size            = 25
beta                  = 0.75
gamma                 = 0.25
lr                    = 0.001
lr_decay_freq         = 30
iters_per_epoch       = training_dataset_size // batch_size

log_freq              = 10
result_pic_freq       = 100

key                   = 'Hell0_World'
hash_algorithm        = 'md5'
key_redundance_size   = image_size // 8
cover_dependent       = False
channel_secret        = 3
channel_cover         = 3
num_downs             = 5
norm_type             = 'batch'
loss                  = 'l2'
