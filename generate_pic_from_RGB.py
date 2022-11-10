from PIL import Image
from torchvision import transforms


def generate_pic(x, y):
    # x = 16  # width    #x坐标  通过对txt里的行数进行整数分解
    # y = 16  # height    #y坐标  x * y = 行数
    im = Image.new("RGB", (x, y), 1)  # 创建图片
    # file = open('2.txt')    #打开rbg值的文件
    # 通过每个rgb点生成图片
    rgb = 255
    for i in range(0, x):
        for j in range(0, y):
            print(rgb)
            im.putpixel((i, j), (rgb, rgb, rgb))  # 将rgb转化为像素
            rgb -= 1
    return im
    # im.save("generate_pic.png")  # im.save('flag.jpg')保存为jpg图片


def get_input(input):
    image = Image.open(input).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze_(0)


if __name__ == "__main__":
    im = Image.new("RGB", (128, 128), color=1)
    # print(im)
    im.save("generate_pic.png")
    # pic_tensor = get_input("generate_pic.png")
    # print(pic_tensor)