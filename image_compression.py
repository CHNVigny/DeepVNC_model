import os
import argparse
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datasets import Datasets, TestKodakDataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter
from PIL import Image
"""
需要两个channel参数，在高质量下是192和320，在低质量下是128和192.
"""



torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 8192
print_freq = 100
cal_step = 40
warmup_step = 0#  // gpu_num
batch_size = 4
tot_epoch = 1000000
tot_step = 2500000
decay_interval = 2200000
lr_decay = 0.1
image_size = 256
logger = logging.getLogger("ImageCompression")
tb_logger = None
global_step = 0
save_model_freq = 50000
test_step = 10000
out_channel_N = 192
out_channel_M = 320

parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')
parser.add_argument('-i', '--input', dest='input', default='high', required=True)
parser.add_argument('-q', '--quality', dest='quality', required=True)
parser.add_argument('-p', '--pretrain', default='', help='load pretrain model', required=True)


def compression_load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    input = Image.open(args.input).convet('RGB')
    model = ImageCompressor(out_channel_N, out_channel_M)
    model = compression_load_model(model, args.pretrain)
    net = model.cuda()
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)
