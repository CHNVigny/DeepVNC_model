import os
import argparse
from model_ksem21 import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datasets import Datasets, TestKodakDataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter
from config import quality_dict

class Decoder():
    def __init__(self, quality):
        self.config = json.load(open(quality_dict[quality]['config']))
        self.pretrain = quality_dict[quality]['model']
        self.gpu_num = torch.cuda.device_count()
        if 'out_channel_N' in self.config:
            self.out_channel_N = self.config['out_channel_N']
        if 'out_channel_M' in self.config:
            self.out_channel_M = self.config['out_channel_M']

    def decode(self, input):
        model_d = ImageCompressor(self.out_channel_N, self.out_channel_M, mod="decoder")
        model_d = load_model(model_d, self.pretrain)
        net_d = model_d.cuda()
        net_d = torch.nn.DataParallel(net_d, list(range(self.gpu_num)))
        net_d.eval()
        image = net_d(input)
        output = image.squeeze(0)
        # print(image.shape)
        return output

if __name__ == "__main__":
    pass
    #dc = Decoder(1).decode(input)

