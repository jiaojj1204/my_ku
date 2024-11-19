import torch
from torchvision import transforms, models
from dataset import data_dataloader
import torch.nn as nn
import torch.optim as optim
from train import train  

def main():
    # 以下是通过Data_dataloader函数输入为：数据的路径，数据大小，batch的大小，有几线并用 （把dataset和Dataloader功能合在了一起）
    train_loader = data_dataloader(data_path='./data/train', size=224, batch_size=24, num_workers=4)
    val_loader = data_dataloader(data_path='./data/val', size=224, batch_size=24, num_workers=2)
    test_loader = data_dataloader(data_path='./data/test', size=224, batch_size=24, num_workers=2)

    # 以下是超参数的定义
    lr = 1e-3           #学习率
    epochs = 20        #训练轮次

    ResNet50 = models.resnet50(pretrained=True)  # 使用预训练模型
    ResNet50.fc = nn.Linear(ResNet50.fc.in_features, 12)  # 修改全连接层
    # 3. 冻结所有层的参数，除了最后一层（fc层）
    for param in ResNet50.parameters():
        param.requires_grad = False  # 冻结所有参数

    # 仅训练最后一层的参数
    for param in ResNet50.fc.parameters():
        param.requires_grad = True  # 解冻最后一层的参数

    optimizer = optim.Adam(ResNet50.parameters(), lr=lr)  # 优化器
    loss_function = nn.CrossEntropyLoss()  # 损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

    # 训练以及验证测试函数
    train(model=ResNet50, optimizer=optimizer, loss_function=loss_function, train_data=train_loader, 
          val_data=val_loader,test_data= test_loader, epochs=epochs, device=device)

if __name__ == '__main__':
    main()