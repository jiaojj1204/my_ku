from dataset import data_dataloader    #电脑本地写的读取数据的函数
from torch import nn                   #导入pytorch的nn模块
from torch import optim                #导入pytorch的optim模块
from network import ResNet50            #电脑本地写的网络框架的函数
from train import train              #电脑本地写的训练函数
import torch                         #导入pytorch

def main():
    # 以下是通过Data_dataloader函数输入为：数据的路径，数据大小，batch的大小，有几线并用 （把dataset和Dataloader功能合在了一起）
    train_loader = data_dataloader(data_path='./data/train', size=224, batch_size=24, num_workers=4)
    val_loader = data_dataloader(data_path='./data/val', size=224, batch_size=24, num_workers=2)
    test_loader = data_dataloader(data_path='./data/test', size=224, batch_size=24, num_workers=2)

    # 以下是超参数的定义
    lr = 1e-3           #学习率
    epochs = 10         #训练轮次

    model = ResNet50(12)  # resnet网络
    # model = torchvision.models.resnet50(pretrained=True)  # 使用预训练模型
    # model.fc = nn.Linear(model.fc.in_features, 12)  # 修改全连接层
    # optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    loss_function = nn.CrossEntropyLoss()  # 损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

    # 训练以及验证测试函数
    train(model=model, optimizer=optimizer, loss_function=loss_function, train_data=train_loader, val_data=val_loader,test_data= test_loader, epochs=epochs, device=device)

if __name__ == '__main__':
    main()