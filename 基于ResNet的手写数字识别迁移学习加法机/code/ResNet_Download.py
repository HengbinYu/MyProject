import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np


image_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数
num_epochs = 20  #训练的总循环周期
batch_size = 64
learning_rate = 0.001

use_cuda = torch.cuda.is_available() #定义一个布尔型变量，标志当前的GPU是否可用

dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

train_dataset = dsets.MNIST(root='./data',  #文件存放路径
                            train=True,   #提取训练集
                            transform=transforms.ToTensor(),  #将图像转化为Tensor
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# 训练数据集的加载器，自动将数据分割成batch，顺序随机打乱
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# 测试数据集的加载器，自动将数据分割成batch
permutes = np.random.permutation(range(len(test_dataset)))
indices_val = permutes[:5000]
indices_test = permutes[5000:]
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)
validation_loader = torch.utils.data.DataLoader(dataset =train_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                sampler = sampler_val
                                               )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          sampler = sampler_test
                                         )

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


def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行10列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = CustomResNet18().to(device)

criterion = nn.CrossEntropyLoss() #Loss函数的定义
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #定义优化器

record = [] #记录准确率等数值的容器
weights = [] #每若干步就记录一次卷积核

# 开始训练循环
for epoch in range(num_epochs):

    train_rights = []  # 记录训练数据集准确率的容器
    for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
        data, target = data.clone().detach().requires_grad_(True), target.clone().detach()  # data为图像，target为标签
        data, target =data.cuda(), target.cuda()
        net.train()  # 给网络模型做标记，标志说模型在训练集上训练
        output = net(data)  # 完成一次预测
        loss = criterion(output, target)  # 计算误差
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 一步随机梯度下降
        loss = loss.cpu() if use_cuda else loss
        right = rightness(output, target)  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
        train_rights.append(right)  # 将计算结果装到列表容器中

        if batch_idx % 100 == 0:  # 每间隔100个batch执行一次

            # train_r为一个二元组，分别记录训练集中分类正确的数量和该集合中总的样本数
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

            net.eval()  # 给网络模型做标记，标志说模型在训练集上训练
            val_rights = []  # 记录校验数据集准确率的容器
            for (data, target) in validation_loader:
                data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
                data, target = data.cuda(), target.cuda()
                output = net(data)  # 完成一次预测
                right = rightness(output, target)  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                val_rights.append(right)

            # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            # 打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
            print('训练周期: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%'.format(
                epoch+1,num_epochs, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].cpu().numpy() / train_r[1],
                       100. * val_r[0].cpu().numpy() / val_r[1]))

            # 将准确率和权重等数值加载到容器中，以方便后续处理
            record.append((100 - 100. * train_r[0].cpu().numpy() / train_r[1], 100 - 100. * val_r[0].cpu().numpy() / val_r[1]))
            # weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
            #                 net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])

# 在这里保存已经训练好的神经网络
torch.save(net, 'Best_ResNet18.pth')