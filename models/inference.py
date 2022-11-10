import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function
from torch.onnx.symbolic_opset9 import ones_like as ol
from torch.onnx.symbolic_opset9 import mul as ml
from torch.onnx.symbolic_opset9 import max as mx
import math
import torch.nn as nn
import torch
#用来导出.pt文件的。



# class LowerBound(Function):
#     @staticmethod
#     def symbolic(g, inputs, bound):
#         ones_like_input_tensor = ol(g, inputs)
#         #print(ones_like_input_tensor)
#         b = ml(g, ones_like_input_tensor, g.op("Constant", value_t=torch.tensor(bound, dtype=float)))
#         return mx(g, inputs, dim_or_y=b, keepdim=None)
#
#     @staticmethod
#     # @torch.jit.script_if_tracing
#     def forward(ctx, inputs, bound):
#         b = torch.ones_like(inputs) * bound
#         ctx.save_for_backward(inputs, b)
#         return torch.max(inputs, b)
#
#     # @staticmethod
#     # def symbolic_backward(g, inputs, bound):
#     #     ones_like_input_tensor = ol(g, inputs)
#     #     # print(ones_like_input_tensor)
#     #     b = ml(g, ones_like_input_tensor, g.op("Constant", value_t=torch.tensor(bound, dtype=float)))
#     #     return mx(g, inputs, dim_or_y=b, keepdim=None)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         inputs, b = ctx.saved_tensors
#         pass_through_1 = inputs >= b
#         pass_through_2 = grad_output < 0
#
#         pass_through = pass_through_1 | pass_through_2
#         return pass_through.type(grad_output.dtype) * grad_output, None
#
#
# class GDN(nn.Module):
#     """Generalized divisive normalization layer.
#     y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
#     """
#
#     def __init__(self,
#                  ch,
#                  inverse=False,
#                  beta_min=1e-6,
#                  gamma_init=0.1,
#                  reparam_offset=2**-18):
#         super(GDN, self).__init__()
#         self.inverse = inverse
#         self.beta_min = beta_min
#         self.gamma_init = gamma_init
#         self.reparam_offset = reparam_offset
#
#         self.build(ch)
#
#     def build(self, ch):
#         self.pedestal = self.reparam_offset**2
#         self.beta_bound = ((self.beta_min + self.reparam_offset**2)**0.5)
#         self.gamma_bound = self.reparam_offset
#
#         # Create beta param
#         beta = torch.sqrt(torch.ones(ch)+self.pedestal)
#         self.beta = nn.Parameter(beta)
#
#         # Create gamma param
#         eye = torch.eye(ch)
#         g = self.gamma_init*eye
#         g = g + self.pedestal
#         gamma = torch.sqrt(g)
#
#         self.gamma = nn.Parameter(gamma)
#         self.pedestal = self.pedestal
#
#     def forward(self, inputs):
#         unfold = False
#         if inputs.dim() == 5:
#             unfold = True
#             bs, ch, d, w, h = inputs.size()
#             inputs = inputs.view(bs, ch, d*w, h)
#
#         _, ch, _, _ = inputs.size()
#
#         # Beta bound and reparam
#         beta = LowerBound.apply(self.beta, self.beta_bound)
#         beta = beta**2 - self.pedestal
#
#         # Gamma bound and reparam
#         gamma = LowerBound.apply(self.gamma, self.gamma_bound)
#         gamma = gamma**2 - self.pedestal
#         gamma = gamma.view(ch, ch, 1, 1)
#
#         # Norm pool calc
#         norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
#         norm_ = torch.sqrt(norm_)
#
#         # Apply norm
#         if self.inverse:
#             outputs = inputs * norm_
#         else:
#             outputs = inputs / norm_
#
#         if unfold:
#             outputs = outputs.view(bs, ch, d, w, h)
#         return outputs

class GDN(nn.Module):
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = ((self.beta_min + self.reparam_offset**2)**0.5) * torch.ones(1)
        self.gamma_bound = self.reparam_offset * torch.ones(1)

        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def low_bound(self, inputs, bound):
        b = torch.ones_like(inputs) * bound
        return torch.max(inputs, b)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        else:
            unfold = False
            bs, ch, w, h = inputs.size()
            d = 1
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = self.low_bound(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = self.low_bound(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        else:#此处做了修改
            return outputs
        return outputs


class Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    """
    todo:把padding改成same试试能不能解决tensor尺寸不对的那个bug。
    """
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis_net, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)

class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Synthesis_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        # self.resDecoder = nn.Sequential(
        #     nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1),# how to initialize ???
        #     GDN(out_channel_N, inverse=True),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),# how to initialize ???
        #     GDN(out_channel_N, inverse=True),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),# how to initialize ???
        #     GDN(out_channel_N, inverse=True),
        #     nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1),# how to initialize ???
        # )

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x


class Bitparm(nn.Module):
    '''
    save params
    '''

    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:#此处做了修改 估计这个没有用
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''

    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

