import torch
import os,glob
import random
import csv
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

# 第一部分：通过三个步骤得到输出的tensor类型的数据
class Dataset_self(Dataset):                    #如果是nn.moduel 则是编写网络模型框架，这里需要继承的是dataset的数据，所以括号中的是Dataset
    #第一步：初始化
    # def __init__(self,root,modo,resize,):       #root是文件根目录，mode是选择什么样的数据集，resize是图像重新调整大小
    def __init__(self, root, resize):       #root是文件根目录，resize是图像    
        super(Dataset_self, self).__init__()
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),  # 调整图像尺寸为224x224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])
        self.root = root
        self.name_label = {}       #创建一个字典来保存每个文件的标签
        #首先得到标签相对于的字典（标签和名称一一对应）
        for name in sorted(os.listdir(os.path.join(root))):     #排序并且用列表的形式打开文件夹
            if not os.path.isdir(os.path.join(root,name)):      #不是文件夹就不需要读取
                continue
            self.name_label[name] = len(self.name_label.keys())  #每个文件的名字为name_Label字典中有多少对键值对的个数
        #print(self.name_label)

        self.image,self.label = self.make_csv('images.csv')       #编写一共函数来读取图片和标签的路径

        #在得到image和label的基础上对图片数据进行一共划分  （注意：如果需要交叉验证就不需要验证集，只划分为训练集和测试集）
        # 有了train、val、test三个数据集了，暂时不需要在一个data里划分了
        # if mode == 'train':
        #     self.image ,self.label= self.image[:int(0.6*len(self.image))],self.label[:int(0.6*len(self.label))]
        # if mode == 'val':
        #     self.image ,self.label= self.image[int(0.6*len(self.image)):int(0.8*len(self.image))],self.label[int(0.6*len(self.label)):int(0.8*len(self.label))]
        # if mode == 'test':
        #     self.image ,self.label= self.image[int(0.8*len(self.image)):],self.label[int(0.8*len(self.label)):]
    
    # 获得图片和标签的函数
    def make_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):  #如果不存在汇总的目录就新建一个
            images = []
            for image in self.name_label.keys():                            # 让image到name_label中的每个文件中去读取图片
                images += glob.glob(os.path.join(self.root, image, '*jpg'))   #加* 贪婪搜索关于jpg的所有文件
            #print('长度为：{}，第二张图片为：{}'.format(len(images),images[1]))
            random.shuffle(images)                                         #把images列表中的数据洗牌
            # images[0]: ./data\ants\382971067_0bfd33afe0.jpg
            with open(os.path.join(self.root,filename), mode='w', newline='') as f :  #创建文件
                writer = csv.writer(f)
                for image in images:
                    name = image.split(os.sep)[-2]  #得到与图片相对应的标签
                    label = self.name_label[name]
                    writer.writerow([image,label])  #写入文件  第一行：./data\ants\382971067_0bfd33afe0.jpg,0
        images,labels = [],[]
        with open(os.path.join(self.root,filename)) as f:   #读取文件
            reader = csv.reader(f)
            for row in reader:
                image, label = row
                label = int(label)
                images.append(image)
                labels.append(label)
        assert len(images) == len(labels)   #类似if语句，只有两者长度一致才继续执行，否则报错
        return images,labels                #返回所有！！是所有的图片和标签（此处的图片不是图片数据本身，而是它的文件目录）

    #第二步：得到图片数据的长度（标签数据长度与图片一致）
    def __len__(self):
        return len(self.image)

    #第三步：读取图片和标签，并输出
    def __getitem__(self, item):   # 单张返回张量的图像与标签
        image,label = self.image[item],self.label[item]      #得到单张图片和相应的标签（此处都是image都是文件目录）
        image = Image.open(image).convert('RGB')             #得到图片数据
        image = self.transform(image)
        label = torch.tensor(label)   #把图片标签也变成tensor类型
        return image,label

#第二部分：使用pytorch自带的DataLoader函数批量得到图片数据
# def data_dataloader(data_path,mode,size,batch_size,num_workers):   #用一个函数加载上诉的数据，data_path、mode和size分别是以上定义的Dataset_self(）中的参数，batch_size是一次性输出多少张图像，num_worker是同时处理几张图像
def data_dataloader(data_path, size, batch_size, num_workers):
    dataset = Dataset_self(data_path, size)
    dataloader = DataLoader(dataset, batch_size, num_workers)  #使用pytorch中的dataloader函数得到数据
    return dataloader

#测试
def main():
    test = Dataset_self('./data/test',64)

if __name__ == '__main__':
    main()