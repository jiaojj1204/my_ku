import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个基本的残差块（Bottleneck）
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1卷积，减少通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3卷积，保持通道数
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1卷积，恢复通道数
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        # 如果存在下采样层，则对输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 顺序通过三个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 将残差连接与主路径相加
        out += identity
        out = self.relu(out)
        
        return out


# ResNet-50模型
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        
        # 初始的7x7卷积层 + 最大池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 定义四个残差块的组
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层，用于分类
        self.fc = nn.Linear(2048, num_classes)

    # 创建每一层残差块
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        # 如果输入和输出通道不同，或者stride为2，需要进行下采样
        if stride != 1 or in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )
        
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * 4
        
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # 按顺序通过各层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


#测试
def main():
    x = torch.randn(2,3,224,224)
    net = ResNet50(12)
    out = net(x)
    print(out.shape)


if __name__ == '__main__':
    main()
