import time
import torch


ngpu = torch.cuda.device_count()
workers = 8
image_size = 128
training_dataset_size = 25000

cur_time = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
ROOT = '/content/drive/My Drive'
DATA_DIR = ROOT + '/dataset'
experiment_dir = ROOT + '/sdh/exper_info/' + cur_time
config_path = experiment_dir + "/config.txt"
log_path = experiment_dir + '/train_log.txt'
checkpoint_path = experiment_dir + '/checkpoint'
train_pics_save_path = experiment_dir + '/train_pics'
val_pics_save_path = experiment_dir + '/val_pics'
test_path = ''
test_pics_save_path = ''
checkpoint = ''
checkpoint_diff = ''

epochs = 65
batch_size = 50
beta = 0.75
lr = 0.001
lr_decay_freq = 30
iters_per_epoch = int(training_dataset_size / batch_size)

log_freq = 10
result_pic_freq = 100

cover_dependent = False
channel_secret = 3
channel_cover = 3
num_downs = 5
norm_type = 'batch'
loss = 'l2'
