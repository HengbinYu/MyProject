# 该文件为训练手写数字识别器的CNN模型并将模型保存下来，以便为后面的与迁移学习做准备

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


image_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数
num_epochs = 20  #训练的总循环周期
batch_size = 64
learning_rate = 0.001

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

# 定义卷积神经网络：4和8为人为指定的两个卷积层的厚度
depth = [4, 8]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)  # 输入通道为1，输出通道为4，窗口大小为5，padding为2
        self.pool = nn.MaxPool2d(2, 2)  # 一个窗口为2*2的pooling运算
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)  # 第二层卷积，输入通道为depth[0], 输出通道为depth[1]，窗口wei15，padding为2
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)  # 一个线性连接层，输入尺寸为最后一层立方体的平铺，输出层512个节点
        self.fc2 = nn.Linear(512, num_classes)  # 最后一层线性分类单元，输入为

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 将立体的Tensor全部转换成一维的Tensor。两次pooling操作，所以图像维度减少了1/4
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        x = F.relu(self.fc1(x))  # 全链接，激活函数
        x = F.dropout(x, training=self.training)  # 以默认为0.5的概率对这一层进行dropout操作
        x = self.fc2(x)  # 全链接，激活函数
        x = F.log_softmax(x, dim=1)  # log_softmax可以理解为概率对数值
        return x

    def retrieve_features(self, x):
        # 提取卷积神经网络的特征图的函数，返回feature_map1, feature_map2为前两层卷积层的特征图
        feature_map1 = F.relu(self.conv1(x))
        x = self.pool(feature_map1)
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1, feature_map2)


def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行10列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels)  # 返回正确的数量和这一次一共比较了多少元素


net = ConvNet()  # 新建一个卷积神经网络的实例

criterion = nn.CrossEntropyLoss()  # Loss函数的定义
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 定义优化器

record = []  # 记录准确率等数值的容器
weights = []  # 每若干步就记录一次卷积核

# 开始训练循环
for epoch in range(num_epochs):

    train_rights = []  # 记录训练数据集准确率的容器
    for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
        data, target = data.clone().detach().requires_grad_(True), target.clone().detach()  # data为图像，target为标签
        net.train()  # 给网络模型做标记，标志说模型在训练集上训练
        output = net(data)  # 完成一次预测
        loss = criterion(output, target)  # 计算误差
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 一步随机梯度下降
        right = rightness(output, target)  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
        train_rights.append(right)  # 将计算结果装到列表容器中

        if batch_idx % 100 == 0:  # 每间隔100个batch执行一次
            # train_r为一个二元组，分别记录训练集中分类正确的数量和该集合中总的样本数
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            net.eval()  # 给网络模型做标记，标志说模型在训练集上训练
            val_rights = []  # 记录校验数据集准确率的容器
            for (data, target) in validation_loader:
                data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
                output = net(data)  # 完成一次预测
                right = rightness(output, target)  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                val_rights.append(right)

            # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            # 打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
            print('训练周期: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].numpy() / train_r[1],
                       100. * val_r[0].numpy() / val_r[1]))

            # 将准确率和权重等数值加载到容器中，以方便后续处理
            record.append((100 - 100. * train_r[0].numpy() / train_r[1], 100 - 100. * val_r[0].numpy() / val_r[1]))
            weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
                            net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])


# 在这里保存已经训练好的神经网络
torch.save(net, 'minst_conv_checkpoint')