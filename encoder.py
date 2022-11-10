import os
import argparse
from model_ksem21 import *
import torch
import json
from PIL import Image
from torchvision import transforms
from config import quality_dict

#quality_dict = {1:{'config':'/examples/example/256.json', 'model':'/checkpoints/256/iter_2660540.pth.tar'}, 2:{'config':'/examples/example/512.json', 'model':'/checkpoints/512/iter_2648162.pth.tar'}, 3:{'config':'/examples/example/1024.json', 'model':'/checkpoints/1024/iter_2660540.pth.tar'}, 4:{'config':'/examples/example/config_low.json', 'model':'/checkpoints/2048/iter_2678432.pth.tar'}, 5:{'config':'/examples/example/4096.json', 'model':'/checkpoints/1024/iter_2660540.pth.tar'}, 6:{'config':'/examples/example/config_high.json', 'model':'/checkpoints/8192/iter_2592648.pth.tar'}}
def get_input(input):
    image = Image.open(input).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)
input_path = "/home/vigny/pic/origin.png"
#gpu_num = torch.cuda.device_count()

# parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')
# parser.add_argument('-n', '--name', default='', help='experiment name')
# parser.add_argument('-p', '--pretrain', default='', help='load pretrain model')
# parser.add_argument('--test', action='store_true')
# parser.add_argument('--config', dest='config', required=False, help='hyperparameter in json format')
# parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')
# parser.add_argument('--train', dest='train', required=True, help='the path of training dataset')
# parser.add_argument('--val', dest='val', required=True, help='the path of validation dataset')
# args = parser.parse_args()

#config = json.load(open(args.config))

# if 'out_channel_N' in config:
#     out_channel_N = config['out_channel_N']
# if 'out_channel_M' in config:
#     out_channel_M = config['out_channel_M']

class Encoder():
    def __init__(self, quality: int):
        #self.quality = quality
        self.config = json.load(open(quality_dict[quality]['config']))
        self.pretrain = quality_dict[quality]['model']
        self.gpu_num = torch.cuda.device_count()
        if 'out_channel_N' in self.config:
            self.out_channel_N = self.config['out_channel_N']
        if 'out_channel_M' in self.config:
            self.out_channel_M = self.config['out_channel_M']
        #self._encode(input)

    def encode(self, input):
        """
        input：一个batch。
        """

        model_e = ImageCompressor(self.out_channel_N, self.out_channel_M, mod="encoder")
        model_e = load_model(model_e, self.pretrain)
        net_e = model_e.cuda()
        net_e = torch.nn.DataParallel(net_e, list(range(self.gpu_num)))
        net_e.eval()#set the model to inference mode.将模型设置为推理模式。.train是将模型设置为训练模式。
        # image_tensor = get_input(input)
        bitstream = net_e(input)
        return bitstream



#如果是训练的话加上net.train()
if __name__ == "__main__":
    print(Encoder(1).encode(get_input(input_path)).size())
    #print(bitstream)
    exit(1)