#torch 1.8.1
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
#input_image = get_input(input_path)
input_image = torch.rand((1, 3, 16, 16))
# print(input_image)
quality = 1
config = json.load(open(quality_dict[quality]['config']))
if 'out_channel_N' in config:
    out_channel_N = config['out_channel_N']
if 'out_channel_M' in config:
    out_channel_M = config['out_channel_M']
    #print("out_channel_M:{0}".format(out_channel_M))
pretrain = quality_dict[quality]['model']
gpu_num = torch.cuda.device_count()
# batch_size = 1
if __name__ == "__main__":
    input_encoder = input_image
    # x = torch.randn(1, 3, 1024, 1024, requires_grad=True)
    # vutils.save_image(x, input_save_path)
    model_e = ImageCompressor(out_channel_N, out_channel_M, mod="encoder")
    model_e = load_model(model_e, pretrain)
    model_e.eval()
    bitstream = model_e(input_encoder)
    # print(bitstream)
    # torch.onnx.export(model_e,  # model being run
    #                   input_encoder,  # model input (or a tuple for multiple inputs)
    #                   "./encoder/onnx10/encoder_256.onnx",
    #                   # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=13,  # the ONNX version to export the model to
    #                   do_constant_folding=False,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   dynamic_axes={'input': {0: 'batch_size', 1: 'channel', 2: 'height', 3: 'width'},
    #                                 # variable length axes
    #                                 'output': {0: 'batch_size', 1: 'channel', 2: 'height', 3: 'width'}})

    input_decoder = bitstream
    model_d = ImageCompressor(out_channel_N, out_channel_M, mod="decoder")
    model_d = load_model(model_d, pretrain)
    # net = model()
    # net = model.cuda()
    # net = torch.nn.DataParallel(net, list(range(gpu_num)))
    model_d.eval()
    output_image = model_d(input_decoder)
    # print(bitstream)
    # torch.onnx.export(model_d,  # model being run
    #                   input_decoder,  # model input (or a tuple for multiple inputs)
    #                   "./decoder/onnx10/decoder_256.onnx",
    #                   # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=13,  # the ONNX version to export the model to
    #                   do_constant_folding=False,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   dynamic_axes={'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},
    #                                 # 这里列出可以动态变化的轴。这里全列出来说明四个量的大小都是动态的。
    #                                 'output': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}})

    print("export completed!")
#vutils.save_image(output_image, output_save_path)
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