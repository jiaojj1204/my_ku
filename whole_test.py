import torch
from PIL import Image
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from new_dataset import data_dataloader 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, mean_squared_error
# 增加 PIL 图像大小限制
Image.MAX_IMAGE_PIXELS = None

# 加载测试数据
class_list = ['00_写实', '01_变形', '02_想象', '03_色彩丰富', '04_色彩对比', '05_线条组合', '06_线条质感', '08_构图能力', '09_转化能力']
test_loader = data_dataloader(data_path='./data/test', size=224, batch_size=24, num_workers=1, classes=class_list)

# 获取类别名称
classes = ['00', '01', '02', '03', '04', '05', '06', '08', '09']
y_true = []
y_pred = []

# 加载预训练的 ResNet50 模型
net = torchvision.models.resnet50(weights=None)
num_ftrs = net.fc.in_features
net.fc = torch.nn.Linear(num_ftrs, 9)  # 修改分类头为9
net.load_state_dict(torch.load('resnet_model_9.pth'))
net.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)  # 将模型移动到设备

# 遍历测试数据
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # 将数据移动到设备
        # 预测
        out = net(images)
        _, preds = torch.max(out, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(len(y_true), len(y_pred))
# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# 绘制混淆矩阵
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.savefig('confusion_matrix.png')

# 计算评估指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


# 计算分类报告
class_report = classification_report(y_true, y_pred, target_names=classes)
print('Classification Report:')
print(class_report)

# 计算平均绝对误差
mae = mean_absolute_error(y_true, y_pred)
print(f'Mean Absolute Error: {mae:.4f}')

# 计算平均平方误差
mse = mean_squared_error(y_true, y_pred)
print(f'Mean Squared Error: {mse:.4f}')