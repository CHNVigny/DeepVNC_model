from PIL import Image
import cDataProcess
# image = Image.open("/home/vigny/pic/origin.jpg").convert('RGB')
# #print(image)
# file = open("data.txt", "rb+")
#
# width = 3
# height = 3
#
# rgb_image = image.convert('RGB')
# # for i in range(0, height):
# # 	for j in range(0, width):
# # 		r, g, b = rgb_image.getpixel((j, i))
# # 		print(r, g, b, file=f)
# # f.close()
# # weight = 1920    #x坐标  通过对txt里的行数进行整数分解
# # height = 1024    #y坐标  x * y = 行数
#
# im = Image.new("RGB", (width, height))   #创建图片
# #file = open('dataTest.txt')    #打开rbg值的文件
#
# #通过每个rgb点生成图片
#
# for k in range(0, height):
#     for m in range(0, width):
#         rgb = []
#         for i in range(0,3):
#             rgb.append(int.from_bytes(file.read(1), 'little', signed=False))#获取一行的rgb值  #分离rgb，文本中逗号后面有空格
#         im.putpixel((m, k), (int(rgb[0]), int(rgb[1]), int(rgb[2])))    #将rgb转化为像素
# #put(width, height)
# im.save('flag.png')
# rgb = cDataProcess.c_data_to_list(3, 3, "data.txt")
# print(rgb)
# im = cDataProcess.c_data_to_image(3, 3, rgb)
# im.save('flag.png')
# exit(0)
#quality_dict = {1:{'config':'/examples/example/256.json', 'model':'/checkpoints/256/iter_2660540.pth.tar'}, 2:{'config':'/examples/example/512.json', 'model':'/checkpoints/512/iter_2648162.pth.tar'}, 3:{'config':'/examples/example/1024.json', 'model':'/checkpoints/1024/iter_2660540.pth.tar'}, 4:{'config':'/examples/example/config_low.json', 'model':'/checkpoints/2048/iter_2678432.pth.tar'}, 5:{'config':'/examples/example/4096.json', 'model':'/checkpoints/1024/iter_2660540.pth.tar'}, 6:{'config':'/examples/example/config_high.json', 'model':'/checkpoints/8192/iter_2592648.pth.tar'}}
from decoder import Decoder
from encoder import Encoder, get_input, input_path
from config import quality_dict
import numpy
import torch
from torchvision import utils as vutils
quality = 1
save_path = "/home/vigny/pic/new_test_1.png"

if __name__ == "__main__":
    en = Encoder(quality)
    de = Decoder(quality)
    input = get_input(input_path)
    output = de.decode(en.encode(input))
    # output = output.clone().detach()
    # output.to(torch.device('cpu'))
    vutils.save_image(output, save_path)


