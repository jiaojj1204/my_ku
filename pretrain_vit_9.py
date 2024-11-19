import torch
from torchvision import transforms, models
from new_dataset import data_dataloader
import torch.nn as nn
import torch.optim as optim
from train import train  

def main():
    class_list = ['00_写实', '01_变形', '02_想象', '03_色彩丰富', '04_色彩对比', '05_线条组合', '06_线条质感', '08_构图能力', '09_转化能力']
    # 以下是通过Data_dataloader函数输入为：数据的路径，数据大小，batch的大小，有几线并用 （把dataset和Dataloader功能合在了一起）
    train_loader = data_dataloader(data_path='./data/train', size=224, batch_size=24, num_workers=4, classes=class_list)
    val_loader = data_dataloader(data_path='./data/val', size=224, batch_size=24, num_workers=2, classes=class_list)
    test_loader = data_dataloader(data_path='./data/test', size=224, batch_size=24, num_workers=2, classes=class_list)

    # 以下是超参数的定义
    lr = 1e-3           #学习率
    epochs = 20        #训练轮次

    # 使用预训练的 Vision Transformer 模型
    ViT = models.vit_b_16(pretrained=True)  # 加载 ViT 模型

    # 检查 heads 模块的类型
    print(ViT.heads)

    # 获取分类头的输入特征维度并替换分类头
    if isinstance(ViT.heads, nn.Sequential):  # 检查是否为 Sequential 模块
        in_features = ViT.heads[0].in_features
    else:
        in_features = ViT.heads.in_features

    ViT.heads = nn.Linear(in_features, 9)  # 修改分类头层以适配 9 类任务

    # 冻结所有层的参数，除了最后一层（分类头层）
    for param in ViT.parameters():
        param.requires_grad = False  # 冻结所有参数

    # 仅训练最后一层的参数
    for param in ViT.heads.parameters():
        param.requires_grad = True  # 解冻分类头层参数

    optimizer = optim.Adam(ViT.parameters(), lr=lr)  # 优化器
    loss_function = nn.CrossEntropyLoss()  # 损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

    # 训练以及验证测试函数
    train(model=ViT, optimizer=optimizer, loss_function=loss_function, train_data=train_loader, 
          val_data=val_loader, test_data=test_loader, epochs=epochs, device=device)

if __name__ == '__main__':
    main()
