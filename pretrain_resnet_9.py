import torch
from torchvision import transforms, models
from new_dataset import data_dataloader
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from new_train import train  

def main():
    class_list = ['0_写实', '1_变形', '2_想象', '3_色彩丰富', '4_色彩对比', '5_线条组合', '6_线条质感', '8_构图能力', '9_转化能力']
    # 以下是通过Data_dataloader函数输入为：数据的路径，数据大小，batch的大小，有几线并用 （把dataset和Dataloader功能合在了一起）
    train_loader = data_dataloader(data_path='./output/train', size=224, batch_size=24, num_workers=4,classes = class_list)
    val_loader = data_dataloader(data_path='./output/val', size=224, batch_size=24, num_workers=2, classes = class_list)
    test_loader = data_dataloader(data_path='./output/test', size=224, batch_size=24, num_workers=2, classes = class_list)

    # 以下是超参数的定义
    lr = 1e-4         #学习率
    epochs = 20        #训练轮次


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    print(device)
    ResNet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # 使用预训练模型
    ResNet50.fc = nn.Linear(ResNet50.fc.in_features, 9)  # 修改全连接层
    ResNet50 = ResNet50.to(device)
    
    # 3. 冻结所有层的参数，除了最后一层（fc层）
    for param in ResNet50.parameters():
        param.requires_grad = False  # 冻结所有参数

    # 仅训练最后一层的参数
    for param in ResNet50.fc.parameters():
        param.requires_grad = True  # 解冻最后一层的参数

    optimizer = optim.Adam(ResNet50.parameters(), lr=lr)  # 优化器
    loss_function = nn.CrossEntropyLoss()  # 损失函数

    
    # 训练以及验证测试函数
    train(model= ResNet50, optimizer=optimizer, loss_function=loss_function, train_loader=train_loader, val_loader=val_loader, 
          test_loader=test_loader, epochs= epochs, device=device, save_dir="./accuary", model_name="new_resnet_9.pth")

if __name__ == '__main__':
    main()


    