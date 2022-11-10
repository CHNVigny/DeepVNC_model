#这个脚本统计各种符号的出现概率。
import io
import numpy as np
from PIL import Image
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
from model_ksem21 import *
from torchvision import transforms
from config import quality_dict
import json
from glob import glob
import collections
import onnxruntime
import onnx
from torchvision import utils as vutils


def get_input(input):
    image = Image.open(input).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze_(0)

input_path = "/home/vigny/pic/origin.png"
input_save_path = "/home/vigny/pic/input_x.png"
output_save_path = "/home/vigny/pic/output_x.png"


class CalcSymbolProb:
    def __init__(self, data_dir, quality, quality_dict, outfile="sym_freq", ):
        self.data_dir = data_dir
        self.quality = quality
        self.total_prob = {}
        self.quality_dict = quality_dict
        self.outfile = "{0}_quality_{1}.json".format(outfile, self.quality)

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))#获取这个文件夹下的所有文件，返回的是一个列表，升序保存。
        # with open(self.outfile, "w") as f:
        #     self.f = f
        try:
            self.f = open(self.outfile, "w")
        except:
            raise RuntimeError("open file error")
        self._load_model(self.quality)
        self.sym_dict = {"quality": self.quality, "statistics_by_file": {}}
        self.total = 0

    def _getitem(self, item):
        # image_ori = self.image_path[item]
        image = Image.open(item).convert('RGB')
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(image).unsqueeze_(0)

    def _load_model(self, quality):
        config = json.load(open(quality_dict[quality]['config']))
        if 'out_channel_N' in config:
            out_channel_N = config['out_channel_N']
        if 'out_channel_M' in config:
            out_channel_M = config['out_channel_M']
            #print("out_channel_M:{0}".format(out_channel_M))
        pretrain = self.quality_dict[quality]['model']
        self.model = ImageCompressor(out_channel_N, out_channel_M, mod="encoder")
        self.model = load_model(self.model, pretrain)
        self.model.eval()
        # return model_e

    def _trans_list(self, bitstream):
        """
        将tensor转换成符合要求的一维列表（list）
        bitstream：输入的张量
        return：转换之后的一维列表
        """
        return (bitstream + 128).byte().reshape(-1).numpy().tolist()


    def _sym_calc(self, bitstream):
        """
        in:tensor
        out:dict, total
        """
        l = self._trans_list(bitstream)
        result = collections.Counter(l)
        return dict(result), sum(result.values())


    def _calc_freq(self, sym_sum, total):
        freq = {}
        for k in sym_sum.keys():
            freq[k] = sym_sum[k] / total
        return freq


    def _gen_filename_dict(self, sym_sum, total, freq):
        d = {}
        d["sum"] = sym_sum
        d["total"] = total
        d["freq"] = freq
        return d

    # def _dump_json(self):
    #     json.dump(self.sym_dict, self.f)
    def _dump_json(self):
        js = json.dumps(self.sym_dict)
        self.f.write(js)

    def start(self):
        statistics_total = {"sum": {}, "total": 0}
        for filename in self.image_path:
            input_tensor = self._getitem(filename)
            bitstream_tensor = self.model(input_tensor)
            sym_sum, total = self._sym_calc(bitstream_tensor)#各个符号出现次数的统计，总的次数
            freq = self._calc_freq(sym_sum, total)
            self.sym_dict["statistics_by_file"][filename] = self._gen_filename_dict(sym_sum, total, freq)
            #计算总量
            for k in sym_sum.keys():
                if k in statistics_total["sum"]:
                    statistics_total["sum"][k] += sym_sum[k]
                else:
                    statistics_total["sum"][k] = sym_sum[k]
            statistics_total["total"] += total

        statistics_total["freq"] = self._calc_freq(statistics_total["sum"], statistics_total["total"])
        self.sym_dict["statistics_total"] = statistics_total
        self._dump_json()
        self.f.close()

    def get_image_path(self):
        return self.image_path






# input_image = get_input(input_path)
# # print(input_image)
# quality = 1
# config = json.load(open(quality_dict[quality]['config']))
# if 'out_channel_N' in config:
#     out_channel_N = config['out_channel_N']
# if 'out_channel_M' in config:
#     out_channel_M = config['out_channel_M']
#     print("out_channel_M:{0}".format(out_channel_M))
# pretrain = quality_dict[quality]['model']
# gpu_num = torch.cuda.device_count()
# batch_size = 1
# input_encoder = input_image
# # x = torch.randn(1, 3, 1024, 1024, requires_grad=True)
# # vutils.save_image(x, input_save_path)
# model_e = ImageCompressor(out_channel_N, out_channel_M, mod="encoder")
# model_e = load_model(model_e, pretrain)
# model_e.eval()
# bitstream = model_e(input_encoder)
# # print(bitstream)
# # print(bitstream.dtype)
#
# l = (bitstream + 128).byte().reshape(-1).numpy().tolist()
#
# # print(l)
# # print(isinstance(l, list))
# d = collections.Counter(l)
# l1 = dict(d)
# s = sum(d.values())
# print(l1)
# print(d)
# print(s)
# j = json.dumps(d)
# with open("test.json", "r") as jf:
#     e = json.load(jf)
#
# print(e["128"])
#
# print(j)

# image_path = sorted(glob(os.path.join("output_path", "*.*")))
# print(len(image_path))

# for c in d:
#        print(f"symbol: {0}, prob: {1}".format(c, d[c]))
#


# print(bitstream)
# torch.onnx.export(model_e,               # model being run
#                   input_encoder,                         # model input (or a tuple for multiple inputs)
#                   "./encoder/onnx10/encoder_256.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=13,          # the ONNX version to export the model to
#                   do_constant_folding=False,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0: 'batch_size', 1: 'channel', 2: 'height', 3: 'width'},    # variable length axes
#                                 'output' : {0: 'batch_size', 1: 'channel', 2: 'height', 3: 'width'}})

# input_decoder = bitstream
# model_d = ImageCompressor(out_channel_N, out_channel_M, mod="decoder")
# model_d = load_model(model_d, pretrain)
# #net = model()
# #net = model.cuda()
# #net = torch.nn.DataParallel(net, list(range(gpu_num)))
# model_d.eval()
# output_image = model_d(input_decoder)
# # print(bitstream)
# torch.onnx.export(model_d,               # model being run
#                   input_decoder,                         # model input (or a tuple for multiple inputs)
#                   "./decoder/onnx10/decoder_256.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=13,          # the ONNX version to export the model to
#                   do_constant_folding=False,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},    #这里列出可以动态变化的轴。这里全列出来说明四个量的大小都是动态的。
#                                 'output' : {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}})
#
# print("export completed!")
# vutils.save_image(output_image, output_save_path)
#
#
#
#
# onnx_model = onnx.load("decoder_256.onnx")
# onnx.checker.check_model(onnx_model)
# ort_session = onnxruntime.InferenceSession("encoder_256.onnx")
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
# ort_outs = ort_session.run(None, ort_inputs)
#
# # print(to_numpy(bitstream).shape)
# # print(np.ndarray(ort_outs).shape)
# # compare ONNX Runtime and PyTorch results
# # if to_numpy(bitstream).size() == ort_outs[0].size():
# #     print(1)
# np.testing.assert_allclose(to_numpy(bitstream), ort_outs[0], rtol=1e-03, atol=1e-05)
#
# print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    for quality in range(4, 7):
        csp = CalcSymbolProb("output_path", quality, quality_dict, outfile="statistic_results/output_path")
        csp.start()

    # l = [1, 1, 2]
    # d = collections.Counter(l)
    # print(sum(d.values()))
