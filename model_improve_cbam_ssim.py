import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *
from torchsummary import summary
from export_onnx import get_input
from adaptive_ac import compress

#尝试在这个模型中加入ms-ssim评价标准

def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        ########
        self.cbam = cbam_block(channel=out_channel_M)
        self.cbam_z = cbam_block(channel=out_channel_N)
        self.ms_ssim = MS_SSIM(data_range=1.0)
        ########
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16).cuda()#
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64).cuda()#生成两个空的张量
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)#两个呈正态分布的张量
        feature = self.Encoder(input_image)
        #print("1:{0}".format(feature.shape))
        #新加入
        feature = self.cbam(feature)
        #print("2:{0}".format(feature.shape))
        #print("cbam")
        #
        batch_size = feature.size()[0]

        z = self.priorEncoder(feature)
        #新加入
        z = self.cbam_z(z)
        #
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        recon_sigma = self.priorDecoder(compressed_z)
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        """算术编码和算数解码在这里"""
        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        # print(recon_image.size())
        # print(input_image.size())
        # exit(3)
        ms_ssim_loss = self.ms_ssim(input_image, clipped_recon_image)
        l_ssim = 1 - ms_ssim_loss
        #print("ssim:{0}".format(l_ssim))
        mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))
        #print("mse:{0}".format(mse_loss))




        def feature_probs_based_sigma(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma) #均值为0，缩放为sigma的拉普拉斯分布。
            # guassian = torch.distributions.normal.Normal(mu, sigma) #normal distribution
            # gaussian = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma) #duoyuan gaosi
            # mix = torch.distributions.Categorical(torch.ones(4, ))
            # gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, gaussian)
            # probs = gmm.cdf(feature + 0.5) - gmm.cdf(feature + 0.5)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            # print(feature)
            # print(probs)
            #print("shape of probs:{0}, shape of feature:{1}".format(probs.shape, feature.shape))
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))#clamp是一个夹紧的操作，并非映射，而是直接改变值。就是把输入重新变成张量了。
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        #print("feature:{0}".format(compressed_feature_renorm.shape))
        total_bits_feature, prob_feature = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        #jiang compressed_feature_renorm zhuan cheng shu zu / transfer compressed_feature_renorm to list
        sym = compressed_feature_renorm.detach() + 128
        sym = sym.reshape(-1).numpy().tolist()
        compress(sym, )
        #print("sym:{0}, feature:{1}".format(sym.shape, compressed_feature_renorm.shape))
        total_bits_z, prob_z = iclr18_estimate_bits_z(compressed_z)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z

        loss = 0.1 * l_ssim + 0.9 * mse_loss

        #loss = mse_loss
        #loss = l_ssim
        return clipped_recon_image, loss, bpp_feature, bpp_z, bpp, ms_ssim_loss, mse_loss


if __name__ == "__main__":
    input = get_input("/home/vigny/pic/origin.png")
    ic = ImageCompressor()
    ic.eval()

    y, loss, bf, bz, b = ic(input)
    #print(loss)
