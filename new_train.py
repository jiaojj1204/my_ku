import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from dataset import Dataset_self
from new_dataset import MyDataset, data_dataloader
from network import ResNet50
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os

# 增加允许的最大像素数
Image.MAX_IMAGE_PIXELS = None

def evaluate(model, loader, device):  
    model.eval()
    correct = 0
    total = len(loader.dataset)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().item()
    model.train()
    return correct / total

def train(model, optimizer, loss_function, train_loader, val_loader, test_loader, 
          epochs, device, save_dir="./accuary", model_name="resnet_model.pth"):
    os.makedirs(save_dir, exist_ok=True)
    best_acc, best_epoch = 0, 0
    train_list, val_list = [], []
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        print(f"============ 第 {epoch + 1} 轮 ============")
        model.train()
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for steps, (x, y) in enumerate(train_loader_tqdm):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_function(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % 10 == 0:  # 每 10 步打印损失
                train_loader_tqdm.set_postfix(loss=loss.item())

        scheduler.step()

        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        train_list.append(train_acc)
        val_list.append(val_acc)

        print(f"train_acc: {train_acc}, val_acc: {val_acc}")

        if val_acc > best_acc:
            best_epoch, best_acc = epoch, val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))

    print(f"Best accuracy: {best_acc} at epoch {best_epoch}")
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name)))
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc}")

    df = pd.DataFrame({'train_acc': train_list, 'val_acc': val_list})
    # 检查文件是否存在并生成新的文件名
    file_index = 1
    file_name = 'accuracy.csv'
    while os.path.exists(file_name):
        file_name = f'accuracy_{file_index}.csv'
        file_index += 1
    df.to_csv(os.path.join(save_dir, file_name), index=False)
