import torch
from torchvision import transforms, models
from new_dataset import data_dataloader
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from new_train import train
from itertools import product

def grid_search(train_loader, val_loader, test_loader, device, class_list):
    # 定义超参数网格
    param_grid = {
        'lr': [1e-4, 1e-3, 1e-5],  # 学习率的候选值
        'batch_size': [16, 24, 32, 48],  # 批次大小的候选值
        'epochs': [10, 15, 20],  # 训练轮次
    }

    best_accuracy = 0  # 最佳准确率初始化
    best_params = {}  # 最佳超参数

    # 使用网格搜索遍历所有超参数组合
    for lr, batch_size, epochs in product(*param_grid.values()):
        print(f"正在尝试组合：学习率={lr}, 批次大小={batch_size}, 轮次={epochs}")
        
        # 重新生成数据加载器，使用当前的批次大小
        current_train_loader = data_dataloader(data_path='./data_select/train', size=224, batch_size=batch_size, num_workers=4, classes=class_list)
        current_val_loader = data_dataloader(data_path='./data_select/val', size=224, batch_size=batch_size, num_workers=2, classes=class_list)
        current_test_loader = data_dataloader(data_path='./data_select/test', size=224, batch_size=batch_size, num_workers=2, classes=class_list)

        # 创建模型
        ResNet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        ResNet50.fc = nn.Linear(ResNet50.fc.in_features, 9)
        ResNet50 = ResNet50.to(device)
        
        # 优化器与损失函数
        optimizer = optim.Adam(ResNet50.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()

        # 训练与验证
        accuracy = train(model=ResNet50, optimizer=optimizer, loss_function=loss_function, 
                         train_loader=current_train_loader, val_loader=current_val_loader, 
                         test_loader=current_test_loader, epochs=epochs, device=device, 
                         save_dir="./accuracy", model_name=f"resnet_grid_search_{lr}_{batch_size}_{epochs}.pth")
        
        # 更新最佳超参数
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'lr': lr, 'batch_size': batch_size, 'epochs': epochs}

    return best_accuracy, best_params


def main():
    class_list = ['0_写实', '1_变形', '2_想象', '3_色彩丰富', '4_色彩对比', '5_线条组合', '6_线条质感', '8_构图能力', '9_转化能力']
    
    # 训练、验证和测试数据加载器
    train_loader = data_dataloader(data_path='./data_select/train', size=224, batch_size=24, num_workers=4, classes=class_list)
    val_loader = data_dataloader(data_path='./data_select/val', size=224, batch_size=24, num_workers=2, classes=class_list)
    test_loader = data_dataloader(data_path='./data_select/test', size=224, batch_size=24, num_workers=2, classes=class_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    # 进行网格搜索
    best_accuracy, best_params = grid_search(train_loader, val_loader, test_loader, device, class_list)

    # 输出最佳超参数组合和对应的准确率
    print(f"最佳准确率：{best_accuracy}")
    print(f"最佳超参数组合：{best_params}")

    # 使用最佳超参数训练最终模型
    final_lr = best_params['lr']
    final_batch_size = best_params['batch_size']
    final_epochs = best_params['epochs']
    
    # 最终模型训练
    ResNet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    ResNet50.fc = nn.Linear(ResNet50.fc.in_features, 9)
    ResNet50 = ResNet50.to(device)
    
    optimizer = optim.Adam(ResNet50.parameters(), lr=final_lr)
    loss_function = nn.CrossEntropyLoss()

    # 训练最终模型
    train(model=ResNet50, optimizer=optimizer, loss_function=loss_function, 
          train_loader=train_loader, val_loader=val_loader, 
          test_loader=test_loader, epochs=final_epochs, device=device, 
          save_dir="./accuracy", model_name="final_resnet_best_model.pth")


if __name__ == '__main__':
    main()
