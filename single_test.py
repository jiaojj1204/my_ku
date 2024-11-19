from network import ResNet50
import torch
from PIL import Image
import torchvision

#导入图片
img = './data/train/0_写实/0b98821ae8796c069a929363e4b1e24f_0_2.jpg'
img =Image.open(img)
tf = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor()])
img = tf(img)
image = torch.reshape(img,(1,3,224,224))
#加载模型
net = ResNet50(12)
net.load_state_dict(torch.load('best.mdl'))
with torch.no_grad():
    out = net(image)
#确定分类
class_cl =out.argmax(dim=1)
class_num = class_cl.numpy()

print(class_num)