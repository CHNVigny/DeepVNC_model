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
import models.inference as mi
from models import *
from torchsummary import summary
#import torchac





def feature_probs_based_sigma(feature, sigma):
    mu = torch.zeros_like(sigma)
    sigma = sigma.clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0,
                                       50))  # clamp是一个夹紧的操作，并非映射，而是直接改变值。这几就是把输入重新变成张量了。
    return total_bits, probs


class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320, mod='training'):
        super(ImageCompressor, self).__init__()
        self.Encoder = mi.Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = mi.Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.bitEstimator_z = mi.BitEstimator(out_channel_N)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M
        self.mod = mod

    def iclr18_estimate_bits_z(self, z):
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
        return total_bits, prob

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)






        if self.mod == "encoder":
            """算术编码在这里"""
            feature = self.Encoder(input_image)
            batch_size = feature.size()[0]
            z = self.priorEncoder(feature)
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
            return compressed_feature_renorm
        elif self.mod == "decoder":
            """算数解码在这里"""
            recon_image = self.Decoder(input_image)
            clipped_recon_image = recon_image.clamp(0., 1.)
            return clipped_recon_image
        elif self.mod == "training":
            feature = self.Encoder(input_image)
            batch_size = feature.size()[0]
            z = self.priorEncoder(feature)
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
            recon_image = self.Decoder(compressed_feature_renorm)
            clipped_recon_image = recon_image.clamp(0., 1.)


            mse_loss = torch.mean((recon_image - input_image).pow(2))



            total_bits_feature, prob_feature = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
            total_bits_z, prob_z = self.iclr18_estimate_bits_z(compressed_z)
            im_shape = input_image.size()
            bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            bpp = bpp_feature + bpp_z
            return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp
        else:
            raise AssertionError('wrong mod')