"""
将来自C语言的数据转换成tensor
"""
from PIL import Image
import torch
from torchvision import transforms

def c_data_to_list(width: int, height: int, file: str)->list:
    """
    file：文件名
    将C语言传递过来的RGB文件按字节读入到一个数组里并返回。
    """
    file = open(file, "rb+")
    rgb = []
    for k in range(0, height):
        row = []
        for m in range(0, width):
            pixel = []
            for i in range(0, 3):
                pixel.append(int.from_bytes(file.read(1), 'little', signed=False))
            row.append(pixel)
        rgb.append(row)
    return rgb
def c_data_list_to_image(width: int, height: int, rgb: list)->Image:
    """
    width：图像的宽
    height：图像的高
    l：图像RGB值的list
    """
    im = Image.new("RGB", (width, height))
    for k in range(0, height):
        for m in range(0, width):
            # 获取一行的rgb值  #分离rgb，文本中逗号后面有空格
            im.putpixel((m, k), (rgb[k][m][0], rgb[k][m][1], rgb[k][m][2]))
    return im

def image_to_tensor(image: Image):
    """
    image：一个image对象
    返回的是一个四维的tensor，因为编码器需要按批处理图像，所以需要把一张图像变成一个size为1的batch。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

if __name__ == "__main__":
    rgb_list = c_data_to_list(4, 3, "data.txt")
    print(rgb_list)
    im = c_data_list_to_image(4, 3, rgb=rgb_list)
    tensor = image_to_tensor(im)
    print(tensor)