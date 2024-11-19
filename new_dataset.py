import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class MyDataset(Dataset):
    def __init__(self, root, transform=None, classes = None):
        self.root = root
        self.transform = transform
        self.classes = classes
        self.dataset = ImageFolder(root=self.root, transform=self.transform)

        if self.classes is not None:
            # 重新分配 class_to_idx
            self.dataset.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            
            # 过滤样本并重新分配标签
            self.dataset.samples = [(path, self.dataset.class_to_idx[os.path.basename(os.path.dirname(path))]) 
                                    for path, _ in self.dataset.samples 
                                    if os.path.basename(os.path.dirname(path)) in self.classes]
            self.dataset.targets = [label for _, label in self.dataset.samples]
        
        self.image = [sample[0] for sample in self.dataset.samples]
        self.label = [sample[1] for sample in self.dataset.samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, torch.tensor(label)

# 使用pytorch自带的DataLoader函数批量得到图片数据
def data_dataloader(data_path, size, batch_size, num_workers, classes = None):
    transform = transforms.Compose([
            transforms.Resize((size, size)),  # 调整图像尺寸为224x224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    dataset = MyDataset(data_path, transform=transform, classes=classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    return dataloader

# 测试
def main():
    class_list = ['00_写实', '01_变形', '02_想象', '03_色彩丰富', '04_色彩对比', '05_线条组合', '06_线条质感', '08_构图能力', '09_转化能力']
    val = MyDataset('./data/val',classes= class_list)
    print(len(val))
    print(val[0])
    print(set(val.label))

if __name__ == '__main__':
    main()