import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import Dataset_self
from new_dataset import MyDataset, data_dataloader
from network import ResNet50
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os

# 增加允许的最大像素数
Image.MAX_IMAGE_PIXELS = None

def evaluate(model,loader,device):   #计算每次训练后的准确率
    model.eval()
    correct = 0
    total = len(loader.dataset)
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)     #得到logits中分类值（要么是[1,0]要么是[0,1]表示分成两个类别）
            correct += torch.eq(pred,y).sum().float().item()        #用logits和标签label想比较得到分类正确的个数
    model.train()
    return correct/total

#把训练的过程定义为一个函数
def train(model,optimizer,loss_function,train_data,val_data,test_data,epochs,device):  #输入：网络架构，优化器，损失函数，训练集，验证集，测试集，轮次
    model = model.to(device) #将模型放入GPU中
    best_acc,best_epoch =0,0      #输出验证集中准确率最高的轮次和准确率
    train_list,val_list = [],[]   # 创建列表保存每一次的acc，用来最后的画图
    for epoch in range(epochs):
            print('============第{}轮============'.format(epoch + 1))
            model.train()
            train_loader = tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs}")
            for steps,(x,y) in enumerate(train_loader): #  for x,y in train_data
                x, y = x.to(device), y.to(device)
                logits = model(x)                   #数据放入网络中
                loss = loss_function(logits,y)      #得到损失值
                optimizer.zero_grad()               #优化器先清零，不然会叠加上次的数值
                loss.backward()                     #后向传播
                optimizer.step()
                train_loader.set_postfix(loss=loss.item())
            
            train_acc =evaluate(model, train_data, device=device)
            train_list.append(train_acc)
            print('train_acc',train_acc)

            #if epoch % 1 == 2:   #这里可以设置每两次训练验证一次
            val_acc = evaluate(model, val_data, device=device)
            print('val_acc=',val_acc)
            val_list.append((val_acc))
            if val_acc > best_acc:  #判断每次在验证集上的准确率是否为最大
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'vit_model_9.pth')   #保存验证集上最大的准确率

    print('===========================分割线===========================')
    print('best acc:',best_acc,'best_epoch:',best_epoch)

    #在测试集上检测训练好后模型的准确率
    model.load_state_dict((torch.load('vit_model_9.pth')))
    print('detect the test data!')
    test_acc = evaluate(model,test_data,device=device)
    print('test_acc:',test_acc)

    # 保存 train_list 和 val_list 到 CSV 文件
    df = pd.DataFrame({'train_acc': train_list, 'val_acc': val_list})

    # 检查文件是否存在并生成新的文件名
    file_index = 1
    file_name = 'accuracy.csv'
    while os.path.exists(file_name):
        file_name = f'accuracy_{file_index}.csv'
        file_index += 1
    df.to_csv(file_name, index=False)

#测试
def main():
    class_list = ['00_写实', '01_变形', '02_想象', '03_色彩丰富', '04_色彩对比', '05_线条组合', '06_线条质感', '08_构图能力', '09_转化能力']
    # 以下是通过Data_dataloader函数输入为：数据的路径，数据大小，batch的大小，有几线并用 （把dataset和Dataloader功能合在了一起）

    train_loader = data_dataloader(data_path='./data/train', size=224, batch_size=24, num_workers=4,classes = class_list)
    val_loader = data_dataloader(data_path='./data/val', size=224, batch_size=24, num_workers=2, classes = class_list)
    test_loader = data_dataloader(data_path='./data/test', size=224, batch_size=24, num_workers=2, classes = class_list)

    lr = 1e-3
    epochs = 5
    model = ResNet50(9)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(model,optimizer,criteon,train_loader,val_loader,test_loader,epochs,device=device)

if __name__ == '__main__':
    main()