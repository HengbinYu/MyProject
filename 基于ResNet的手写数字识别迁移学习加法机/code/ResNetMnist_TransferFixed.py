# 该文件为使用ResNet18进行手写数字加法器的模型与训练脚本代码，模式为固定值迁移学习

import pandas as pd
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import openpyxl

FigPath='D:\\work\\PythonProject\\DeepLearning\\Study\\手写数字识别\\06_Transfer Learning\\figs\\TransferFixed\\'
FilePath='D:\\work\\PythonProject\\DeepLearning\\Study\\手写数字识别\\06_Transfer Learning\\results\\Result_TransferFixed.xlsx'

# 设置图像读取器的超参数
image_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数
num_epochs = 10  #训练的总循环周期
batch_size = 64  #批处理的尺寸大小
use_cuda = torch.cuda.is_available() #定义一个布尔型变量，标志当前的GPU是否可用

dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

train_dataset = dsets.MNIST(root='./data',  #文件存放路径
                            train=True,   #提取训练集
                            transform=transforms.ToTensor(),  #将图像转化为Tensor
                            download=True)

# 加载测试数据集
test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())


# 定义两个采样器，每一个采样器都随机地从原始的数据集中抽样数据。抽样数据采用permutation
# 生成任意一个下标重排，从而利用下标来提取dataset中的数据
sample_size = len(train_dataset)
sampler1 = torch.utils.data.sampler.SubsetRandomSampler(
    np.random.choice(range(len(train_dataset)), sample_size))
sampler2 = torch.utils.data.sampler.SubsetRandomSampler(
    np.random.choice(range(len(train_dataset)), sample_size))

# 定义两个加载器，分别封装了前两个采样器，实现采样。
train_loader1 = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           sampler = sampler1
                                           )
train_loader2 = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           sampler = sampler2
                                           )

# 对于校验数据和测试数据，进行类似的处理。
val_size = 5000
val_indices1 = range(val_size)
val_indices2 = np.random.permutation(range(val_size))
test_indices1 = range(val_size, len(test_dataset))#将剩下数据进行划分，前面是验证集，后面是测试集
test_indices2 = np.random.permutation(test_indices1)
val_sampler1 = torch.utils.data.sampler.SubsetRandomSampler(val_indices1)
val_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(val_indices2)

test_sampler1 = torch.utils.data.sampler.SubsetRandomSampler(test_indices1)
test_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(test_indices2)

val_loader1 = torch.utils.data.DataLoader(dataset = test_dataset,
                                        batch_size = batch_size,
                                        shuffle = False,
                                        sampler = val_sampler1
                                        )
val_loader2 = torch.utils.data.DataLoader(dataset = test_dataset,
                                        batch_size = batch_size,
                                        shuffle = False,
                                        sampler = val_sampler2
                                        )
test_loader1 = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False,
                                         sampler = test_sampler1
                                         )
test_loader2 = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False,
                                         sampler = test_sampler2
                                         )

# 加载网络
import torchvision.models as models
class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.Flatten = nn.Flatten()
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = self.resnet18.Flatten(x)
        x = self.resnet18.fc(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
custom_resnet18= torch.load('Best_ResNet18.pth')
print(custom_resnet18)


class Transfer_ResNet(nn.Module):
    def __init__(self):
        super(Transfer_ResNet, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18A = models.resnet18(weights=None)
        self.resnet18B = models.resnet18(weights=None)
        # 两个并行的卷积通道:
        self.resnet18A.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为1
        self.resnet18B.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为1
        self.resnet18.Flatten = nn.Flatten()
        num_ftrs = 2 * self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 19)

    def forward(self, x, y, training=True):
        # 首先，第一张图像进入第一个通道
        x = F.relu(self.resnet18A.bn1(self.resnet18A.conv1(x)))  # 第一层卷积
        x = self.resnet18A.maxpool(x)  # 第一层池化
        x = self.resnet18A.layer1(x)
        x = self.resnet18A.layer2(x)
        x = self.resnet18A.layer3(x)
        x = self.resnet18A.layer4(x)
        x = self.resnet18A.avgpool(x)
        x = self.resnet18.Flatten(x)

        # 第2张图像进入第2个通道
        y = F.relu(self.resnet18B.bn1(self.resnet18B.conv1(y)))  # 第一层卷积
        y = self.resnet18B.maxpool(y)  # 第一层池化
        y = self.resnet18B.layer1(y)
        y = self.resnet18B.layer2(y)
        y = self.resnet18B.layer3(y)
        y = self.resnet18B.layer4(y)
        y = self.resnet18B.avgpool(y)
        y = self.resnet18.Flatten(y)

        # 将两个卷积过来的铺平向量拼接在一起，形成一个大向量
        z = torch.cat((x, y), 1)  # cat函数为拼接向量操作，1表示拼接的维度为第1个维度（0维度对应了batch）
        z = self.resnet18.fc(z)
        return z

    def set_filter_values(self, net):
        self.resnet18A.conv1.weight.data = copy.deepcopy(net.resnet18.conv1.weight.data)
        self.resnet18B.conv1.weight.data = copy.deepcopy(net.resnet18.conv1.weight.data)

        # 第一层
        # A：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18A.layer1[0].conv1.weight.data = copy.deepcopy(net.resnet18.layer1[0].conv1.weight.data)
        # self.resnet18A.layer1[0].conv1.bias.data = copy.deepcopy(net.resnet18.layer1[0].conv1.bias.data)
        self.resnet18A.layer1[0].conv2.weight.data = copy.deepcopy(net.resnet18.layer1[0].conv2.weight.data)
        # self.resnet18A.layer1[0].conv2.bias.data = copy.deepcopy(net.resnet18.layer1[0].conv2.bias.data)
        # A :第一层BasicBlock1
        self.resnet18A.layer1[1].conv1.weight.data = copy.deepcopy(net.resnet18.layer1[1].conv1.weight.data)
        # self.resnet18A.layer1[1].conv1.bias.data = copy.deepcopy(net.resnet18.layer1[1].conv1.bias.data)
        self.resnet18A.layer1[1].conv2.weight.data = copy.deepcopy(net.resnet18.layer1[1].conv2.weight.data)
        # self.resnet18A.layer1[1].conv2.bias.data = copy.deepcopy(net.resnet18.layer1[1].conv2.bias.data)
        # B：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18B.layer1[0].conv1.weight.data = copy.deepcopy(net.resnet18.layer1[0].conv1.weight.data)
        # self.resnet18B.layer1[0].conv1.bias.data = copy.deepcopy(net.resnet18.layer1[0].conv1.bias.data)
        self.resnet18B.layer1[0].conv2.weight.data = copy.deepcopy(net.resnet18.layer1[0].conv2.weight.data)
        # self.resnet18B.layer1[0].conv2.bias.data = copy.deepcopy(net.resnet18.layer1[0].conv2.bias.data)
        # B :第一层BasicBlock1
        self.resnet18B.layer1[1].conv1.weight.data = copy.deepcopy(net.resnet18.layer1[1].conv1.weight.data)
        # self.resnet18B.layer1[1].conv1.bias.data = copy.deepcopy(net.resnet18.layer1[1].conv1.bias.data)
        self.resnet18B.layer1[1].conv2.weight.data = copy.deepcopy(net.resnet18.layer1[1].conv2.weight.data)
        # self.resnet18B.layer1[1].conv2.bias.data = copy.deepcopy(net.resnet18.layer1[1].conv2.bias.data)

        # 第二层
        # A：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18A.layer2[0].conv1.weight.data = copy.deepcopy(net.resnet18.layer2[0].conv1.weight.data)
        # self.resnet18A.layer2[0].conv1.bias.data = copy.deepcopy(net.resnet18.layer2[0].conv1.bias.data)
        self.resnet18A.layer2[0].conv2.weight.data = copy.deepcopy(net.resnet18.layer2[0].conv2.weight.data)
        # self.resnet18A.layer2[0].conv2.bias.data = copy.deepcopy(net.resnet18.layer2[0].conv2.bias.data)
        # A :第一层BasicBlock1
        self.resnet18A.layer2[1].conv1.weight.data = copy.deepcopy(net.resnet18.layer2[1].conv1.weight.data)
        # self.resnet18A.layer2[1].conv1.bias.data = copy.deepcopy(net.resnet18.layer2[1].conv1.bias.data)
        self.resnet18A.layer2[1].conv2.weight.data = copy.deepcopy(net.resnet18.layer2[1].conv2.weight.data)
        # self.resnet18A.layer2[1].conv2.bias.data = copy.deepcopy(net.resnet18.layer2[1].conv2.bias.data)
        # B：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18B.layer2[0].conv1.weight.data = copy.deepcopy(net.resnet18.layer2[0].conv1.weight.data)
        # self.resnet18B.layer2[0].conv1.bias.data = copy.deepcopy(net.resnet18.layer2[0].conv1.bias.data)
        self.resnet18B.layer2[0].conv2.weight.data = copy.deepcopy(net.resnet18.layer2[0].conv2.weight.data)
        # self.resnet18B.layer2[0].conv2.bias.data = copy.deepcopy(net.resnet18.layer2[0].conv2.bias.data)
        # B :第一层BasicBlock1
        self.resnet18B.layer2[1].conv1.weight.data = copy.deepcopy(net.resnet18.layer2[1].conv1.weight.data)
        # self.resnet18B.layer2[1].conv1.bias.data = copy.deepcopy(net.resnet18.layer2[1].conv1.bias.data)
        self.resnet18B.layer2[1].conv2.weight.data = copy.deepcopy(net.resnet18.layer2[1].conv2.weight.data)
        # self.resnet18B.layer2[1].conv2.bias.data = copy.deepcopy(net.resnet18.layer2[1].conv2.bias.data)

        # 第三层
        # A：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18A.layer3[0].conv1.weight.data = copy.deepcopy(net.resnet18.layer3[0].conv1.weight.data)
        # self.resnet18A.layer3[0].conv1.bias.data = copy.deepcopy(net.resnet18.layer3[0].conv1.bias.data)
        self.resnet18A.layer3[0].conv2.weight.data = copy.deepcopy(net.resnet18.layer3[0].conv2.weight.data)
        # self.resnet18A.layer3[0].conv2.bias.data = copy.deepcopy(net.resnet18.layer3[0].conv2.bias.data)
        # A :第一层BasicBlock1
        self.resnet18A.layer3[1].conv1.weight.data = copy.deepcopy(net.resnet18.layer3[1].conv1.weight.data)
        # self.resnet18A.layer3[1].conv1.bias.data = copy.deepcopy(net.resnet18.layer3[1].conv1.bias.data)
        self.resnet18A.layer3[1].conv2.weight.data = copy.deepcopy(net.resnet18.layer3[1].conv2.weight.data)
        # self.resnet18A.layer3[1].conv2.bias.data = copy.deepcopy(net.resnet18.layer3[1].conv2.bias.data)
        # B：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18B.layer3[0].conv1.weight.data = copy.deepcopy(net.resnet18.layer3[0].conv1.weight.data)
        # self.resnet18B.layer3[0].conv1.bias.data = copy.deepcopy(net.resnet18.layer3[0].conv1.bias.data)
        self.resnet18B.layer3[0].conv2.weight.data = copy.deepcopy(net.resnet18.layer3[0].conv2.weight.data)
        # self.resnet18B.layer3[0].conv2.bias.data = copy.deepcopy(net.resnet18.layer3[0].conv2.bias.data)
        # B :第一层BasicBlock1
        self.resnet18B.layer3[1].conv1.weight.data = copy.deepcopy(net.resnet18.layer3[1].conv1.weight.data)
        # self.resnet18B.layer3[1].conv1.bias.data = copy.deepcopy(net.resnet18.layer3[1].conv1.bias.data)
        self.resnet18B.layer3[1].conv2.weight.data = copy.deepcopy(net.resnet18.layer3[1].conv2.weight.data)
        # self.resnet18B.layer3[1].conv2.bias.data = copy.deepcopy(net.resnet18.layer3[1].conv2.bias.data)

        # 第四层
        # A：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18A.layer4[0].conv1.weight.data = copy.deepcopy(net.resnet18.layer4[0].conv1.weight.data)
        # self.resnet18A.layer4[0].conv1.bias.data = copy.deepcopy(net.resnet18.layer4[0].conv1.bias.data)
        self.resnet18A.layer4[0].conv2.weight.data = copy.deepcopy(net.resnet18.layer4[0].conv2.weight.data)
        # self.resnet18A.layer4[0].conv2.bias.data = copy.deepcopy(net.resnet18.layer4[0].conv2.bias.data)
        # A :第一层BasicBlock1
        self.resnet18A.layer4[1].conv1.weight.data = copy.deepcopy(net.resnet18.layer4[1].conv1.weight.data)
        # self.resnet18A.layer4[1].conv1.bias.data = copy.deepcopy(net.resnet18.layer4[1].conv1.bias.data)
        self.resnet18A.layer4[1].conv2.weight.data = copy.deepcopy(net.resnet18.layer4[1].conv2.weight.data)
        # self.resnet18A.layer4[1].conv2.bias.data = copy.deepcopy(net.resnet18.layer4[1].conv2.bias.data)
        # B：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18B.layer4[0].conv1.weight.data = copy.deepcopy(net.resnet18.layer4[0].conv1.weight.data)
        # self.resnet18B.layer4[0].conv1.bias.data = copy.deepcopy(net.resnet18.layer4[0].conv1.bias.data)
        self.resnet18B.layer4[0].conv2.weight.data = copy.deepcopy(net.resnet18.layer4[0].conv2.weight.data)
        # self.resnet18B.layer4[0].conv2.bias.data = copy.deepcopy(net.resnet18.layer4[0].conv2.bias.data)
        # B :第一层BasicBlock1
        self.resnet18B.layer4[1].conv1.weight.data = copy.deepcopy(net.resnet18.layer4[1].conv1.weight.data)
        # self.resnet18B.layer4[1].conv1.bias.data = copy.deepcopy(net.resnet18.layer4[1].conv1.bias.data)
        self.resnet18B.layer4[1].conv2.weight.data = copy.deepcopy(net.resnet18.layer4[1].conv2.weight.data)
        # self.resnet18B.layer4[1].conv2.bias.data = copy.deepcopy(net.resnet18.layer4[1].conv2.bias.data)

        # 将模型加载到GPU上
        self.resnet18A.conv1 = self.resnet18A.conv1.cuda() if use_cuda else self.resnet18A.conv1
        self.resnet18B.conv1 = self.resnet18B.conv1.cuda() if use_cuda else self.resnet18B.conv1

        self.resnet18A.layer1[0].conv1 = self.resnet18A.layer1[0].conv1.cuda() if use_cuda else self.resnet18A.layer1[0].conv1
        self.resnet18A.layer1[0].conv2 = self.resnet18A.layer1[0].conv2.cuda() if use_cuda else self.resnet18A.layer1[0].conv2
        self.resnet18A.layer1[1].conv1 = self.resnet18A.layer1[1].conv1.cuda() if use_cuda else self.resnet18A.layer1[1].conv1
        self.resnet18A.layer1[1].conv2 = self.resnet18A.layer1[1].conv2.cuda() if use_cuda else self.resnet18A.layer1[1].conv2
        self.resnet18B.layer1[0].conv1 = self.resnet18B.layer1[0].conv1.cuda() if use_cuda else self.resnet18B.layer1[0].conv1
        self.resnet18B.layer1[0].conv2 = self.resnet18B.layer1[0].conv2.cuda() if use_cuda else self.resnet18B.layer1[0].conv2
        self.resnet18B.layer1[1].conv1 = self.resnet18B.layer1[1].conv1.cuda() if use_cuda else self.resnet18B.layer1[1].conv1
        self.resnet18B.layer1[1].conv2 = self.resnet18B.layer1[1].conv2.cuda() if use_cuda else self.resnet18B.layer1[1].conv2
        
        self.resnet18A.layer2[0].conv1 = self.resnet18A.layer2[0].conv1.cuda() if use_cuda else self.resnet18A.layer2[0].conv1
        self.resnet18A.layer2[0].conv2 = self.resnet18A.layer2[0].conv2.cuda() if use_cuda else self.resnet18A.layer2[0].conv2
        self.resnet18A.layer2[1].conv1 = self.resnet18A.layer2[1].conv1.cuda() if use_cuda else self.resnet18A.layer2[1].conv1
        self.resnet18A.layer2[1].conv2 = self.resnet18A.layer2[1].conv2.cuda() if use_cuda else self.resnet18A.layer2[1].conv2
        self.resnet18B.layer2[0].conv1 = self.resnet18B.layer2[0].conv1.cuda() if use_cuda else self.resnet18B.layer2[0].conv1
        self.resnet18B.layer2[0].conv2 = self.resnet18B.layer2[0].conv2.cuda() if use_cuda else self.resnet18B.layer2[0].conv2
        self.resnet18B.layer2[1].conv1 = self.resnet18B.layer2[1].conv1.cuda() if use_cuda else self.resnet18B.layer2[1].conv1
        self.resnet18B.layer2[1].conv2 = self.resnet18B.layer2[1].conv2.cuda() if use_cuda else self.resnet18B.layer2[1].conv2
        
        self.resnet18A.layer3[0].conv1 = self.resnet18A.layer3[0].conv1.cuda() if use_cuda else self.resnet18A.layer3[0].conv1
        self.resnet18A.layer3[0].conv2 = self.resnet18A.layer3[0].conv2.cuda() if use_cuda else self.resnet18A.layer3[0].conv2
        self.resnet18A.layer3[1].conv1 = self.resnet18A.layer3[1].conv1.cuda() if use_cuda else self.resnet18A.layer3[1].conv1
        self.resnet18A.layer3[1].conv2 = self.resnet18A.layer3[1].conv2.cuda() if use_cuda else self.resnet18A.layer3[1].conv2
        self.resnet18B.layer3[0].conv1 = self.resnet18B.layer3[0].conv1.cuda() if use_cuda else self.resnet18B.layer3[0].conv1
        self.resnet18B.layer3[0].conv2 = self.resnet18B.layer3[0].conv2.cuda() if use_cuda else self.resnet18B.layer3[0].conv2
        self.resnet18B.layer3[1].conv1 = self.resnet18B.layer3[1].conv1.cuda() if use_cuda else self.resnet18B.layer3[1].conv1
        self.resnet18B.layer3[1].conv2 = self.resnet18B.layer3[1].conv2.cuda() if use_cuda else self.resnet18B.layer3[1].conv2
        
        self.resnet18A.layer4[0].conv1 = self.resnet18A.layer4[0].conv1.cuda() if use_cuda else self.resnet18A.layer4[0].conv1
        self.resnet18A.layer4[0].conv2 = self.resnet18A.layer4[0].conv2.cuda() if use_cuda else self.resnet18A.layer4[0].conv2
        self.resnet18A.layer4[1].conv1 = self.resnet18A.layer4[1].conv1.cuda() if use_cuda else self.resnet18A.layer4[1].conv1
        self.resnet18A.layer4[1].conv2 = self.resnet18A.layer4[1].conv2.cuda() if use_cuda else self.resnet18A.layer4[1].conv2
        self.resnet18B.layer4[0].conv1 = self.resnet18B.layer4[0].conv1.cuda() if use_cuda else self.resnet18B.layer4[0].conv1
        self.resnet18B.layer4[0].conv2 = self.resnet18B.layer4[0].conv2.cuda() if use_cuda else self.resnet18B.layer4[0].conv2
        self.resnet18B.layer4[1].conv1 = self.resnet18B.layer4[1].conv1.cuda() if use_cuda else self.resnet18B.layer4[1].conv1
        self.resnet18B.layer4[1].conv2 = self.resnet18B.layer4[1].conv2.cuda() if use_cuda else self.resnet18B.layer4[1].conv2
    def set_filter_values_nograd(self, net):
        # 本函数为迁移网络所用，即将迁移过来的网络的权重值拷贝到本网络中
        # 本函数对应的迁移为固定权重式
        # 调用set_filter_values为全部卷积核进行赋值
        self.set_filter_values(net)
        self.resnet18A.conv1.weight.requires_grad = False
        self.resnet18B.conv1.weight.requires_grad = False

        # 第一层
        # A：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18A.layer1[0].conv1.weight.requires_grad = False
        self.resnet18A.layer1[0].conv2.weight.requires_grad = False
        # A :第一层BasicBlock1
        self.resnet18A.layer1[1].conv1.weight.requires_grad = False
        self.resnet18A.layer1[1].conv2.weight.requires_grad = False
        # B：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18B.layer1[0].conv1.weight.requires_grad = False
        self.resnet18B.layer1[0].conv2.weight.requires_grad = False
        # B :第一层BasicBlock1
        self.resnet18B.layer1[1].conv1.weight.requires_grad = False
        self.resnet18B.layer1[1].conv2.weight.requires_grad = False

        # 第二层
        # A：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18A.layer2[0].conv1.weight.requires_grad = False
        self.resnet18A.layer2[0].conv2.weight.requires_grad = False
        # A :第一层BasicBlock1
        self.resnet18A.layer2[1].conv1.weight.requires_grad = False
        self.resnet18A.layer2[1].conv2.weight.requires_grad = False
        # B：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18B.layer2[0].conv1.weight.requires_grad = False
        self.resnet18B.layer2[0].conv2.weight.requires_grad = False
        # B :第一层BasicBlock1
        self.resnet18B.layer2[1].conv1.weight.requires_grad = False
        self.resnet18B.layer2[1].conv2.weight.requires_grad = False

        # 第三层
        # A：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18A.layer3[0].conv1.weight.requires_grad = False
        self.resnet18A.layer3[0].conv2.weight.requires_grad = False
        # A :第一层BasicBlock1
        self.resnet18A.layer3[1].conv1.weight.requires_grad = False
        self.resnet18A.layer3[1].conv2.weight.requires_grad = False
        # B：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18B.layer3[0].conv1.weight.requires_grad = False
        self.resnet18B.layer3[0].conv2.weight.requires_grad = False
        # B :第一层BasicBlock1
        self.resnet18B.layer3[1].conv1.weight.requires_grad = False
        self.resnet18B.layer3[1].conv2.weight.requires_grad = False

        # 第四层
        # A：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18A.layer4[0].conv1.weight.requires_grad = False
        self.resnet18A.layer4[0].conv2.weight.requires_grad = False
        # A :第一层BasicBlock1
        self.resnet18A.layer4[1].conv1.weight.requires_grad = False
        self.resnet18A.layer4[1].conv2.weight.requires_grad = False
        # B：第一层BasicBlock0：两个卷积，每个卷积一个权重一个偏置
        self.resnet18B.layer4[0].conv1.weight.requires_grad = False
        self.resnet18B.layer4[0].conv2.weight.requires_grad = False
        # B :第一层BasicBlock1
        self.resnet18B.layer4[1].conv1.weight.requires_grad = False
        self.resnet18B.layer4[1].conv2.weight.requires_grad = False


def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行10列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels)  # 返回正确的数量和这一次一共比较了多少元素


# 为了比较不同数据量对迁移学习的影响，设定了一个加载数据的比例fraction
# 即只加载原训练数据集的1/fraction来训练网络
fractions = [20, 10, 8, 6, 5, 4, 3, 2, 1]
# fractions = [50, 30]
with pd.ExcelWriter(FilePath) as writer:
    for fraction in fractions:
        # 生成网络实例
        net = Transfer_ResNet().to(device)
        net.set_filter_values_nograd(custom_resnet18)
        if use_cuda:
            net = net.cuda()
        print(use_cuda)
        # 定义损失函数，我们用最小均方误差来定义损失
        criterion = nn.CrossEntropyLoss()
        # 将需要训练的参数加载到优化器中
        new_parameters = []
        for para in net.parameters():
            if para.requires_grad:  # 只将可以调整权重的变量加到了集合new_parameters
                new_parameters.append(para)

        # 将new_parameters加载到了优化器中
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # 开始训练网络
        num_epochs = 10
        records = []
        start_time=time.time()
        for epoch in range(num_epochs):
            losses = []
            train_rights = []
            for idx, data in enumerate(zip(train_loader1, train_loader2)):
                # 为了比较数据量大小对迁移学习的影响，只加载了部分数据，当数据加载个数idx大于
                # 全数据集的1/fraction的时候则不再加载后面的数据
                if idx >= (len(train_loader1) // fraction):
                    break
                ((x1, y1), (x2, y2)) = data
                if use_cuda:
                    x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
                net.train()
                outputs = net(x1.clone().detach().requires_grad_(True), x2.clone().detach().requires_grad_(True))
                outputs = outputs.squeeze()
                labels = y1 + y2
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                right_train = rightness(outputs, labels)
                train_rights.append(right_train)
                loss = loss.cpu() if use_cuda else loss
                losses.append(loss.data.numpy())
                if idx % 10 == 0:
                    # 在校验数据上计算计算准确率
                    val_losses = []
                    val_rights = []
                    net.eval()
                    for val_data in zip(val_loader1, val_loader2):
                        ((x1, y1), (x2, y2)) = val_data
                        if use_cuda:
                            x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
                        outputs = net(x1.clone().detach().requires_grad_(True), x2.clone().detach().requires_grad_(True))
                        outputs = outputs.squeeze()
                        labels = y1 + y2
                        loss = criterion(outputs, labels)
                        loss = loss.cpu() if use_cuda else loss
                        val_losses.append(loss.data.numpy())
                        right_val = rightness(outputs, labels)
                        val_rights.append(right_val)

                    train_right_ratio = 1.0 * np.sum([i[0].cpu().numpy() for i in train_rights]) / np.sum([i[1] for i in train_rights])
                    val_right_ratio = 1.0 * np.sum([i[0].cpu().numpy() for i in val_rights]) / np.sum([i[1] for i in val_rights])
                    print('第{}周期，第({}/{})个撮，训练误差：{}，训练正确率：{:.2f}, 校验误差：{:.2f}, 校验正确率：{:.2f}'.format(
                        epoch+1, idx, len(train_loader1),np.mean(losses),train_right_ratio, np.mean(val_losses), val_right_ratio))
                    records.append([np.mean(losses), np.mean(val_losses), train_right_ratio,val_right_ratio])
        end_time=time.time()
        # 在测试数据集上测试准确度
        test_rights = []
        net.eval()
        for test_data in zip(test_loader1, test_loader2):
            ((x1, y1), (x2, y2)) = test_data
            if use_cuda:
                x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
            outputs = net(x1.clone().detach().requires_grad_(True), x2.clone().detach().requires_grad_(True))
            outputs = outputs.squeeze()
            labels = y1 + y2
            loss = criterion(outputs, labels)
            right_test = rightness(outputs, labels)
            test_rights.append(right_test)
        test_right_ratio = 1.0 * np.sum([i[0].cpu().numpy() for i in test_rights]) / np.sum([i[1] for i in test_rights])
        time_cost=end_time-start_time
        print('测试集正确率：{}'.format(test_right_ratio))
        print('训练耗时：{}'.format(time_cost))

        Train_loss=pd.Series(name='Train_loss')
        Train_right=pd.Series(name='Train_right')
        Val_loss=pd.Series(name='Val_loss')
        Val_right=pd.Series(name='Val_right')
        Test_right=pd.Series(name='Test_right')
        Time_cost=pd.Series(name='Time_cost')

        for i,index in enumerate(records):
            Train_loss[i]=index[0]
            Val_loss[i]=index[1]
            Train_right[i]=index[2]
            Val_right[i]=index[3]
            if i==0:
                Test_right[i]=test_right_ratio
                Time_cost[i]=time_cost
            else:
                Test_right[i]=None
                Time_cost[i]=None

        df=pd.concat([Train_loss,Val_loss,Train_right,Val_right,Test_right,Time_cost], axis=1)
        SheetName='Fraction={}'.format(fraction)
        df.to_excel(writer, sheet_name=SheetName)

        # 画图
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df.index, df['Train_loss'], label='Train_loss', marker='.')
        ax1.plot(df.index, df['Val_loss'], label='Val_loss', marker='*')
        ax1.set_xlabel('Epoch')
        ax1.set_ylim(0, 3.5)
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['Train_right'], label='Train_right', marker='.', linestyle='--', color='green')
        ax2.plot(df.index, df['Val_right'], label='Val_right', marker='*', linestyle='--', color='red')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='upper right')

        plt.title('TransferFixed: fraction={} ; Test_right={:.2%}, Time_cost={:.2f}s'.format(fraction,test_right_ratio,time_cost))
        SaveFigPath=FigPath+'TransferFixed_fraction_{}'.format(fraction)+'.png'
        plt.savefig(SaveFigPath,dpi=1200)
        plt.show()
        print('{}:写入保存完成'.format(SheetName))