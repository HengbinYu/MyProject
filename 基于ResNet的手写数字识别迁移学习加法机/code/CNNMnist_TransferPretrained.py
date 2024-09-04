# 该文件为使用CNN进行手写数字加法器的模型与训练脚本代码，模式为预训练迁移学习

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

FigPath='D:\\work\\PythonProject\\DeepLearning\\Study\\手写数字识别\\06_Transfer Learning\\figs\\CNN_TransferPretrained\\'
FilePath='D:\\work\\PythonProject\\DeepLearning\\Study\\手写数字识别\\06_Transfer Learning\\results\\Result_CNN_TransferPretrained.xlsx'

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
depth = [4, 8]
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 将立体的Tensor全部转换成一维的Tensor。两次pooling操作，所以图像维度减少了1/4
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        x = F.relu(self.fc1(x)) #全链接，激活函数
        x = F.dropout(x, training=self.training) #以默认为0.5的概率对这一层进行dropout操作
        x = self.fc2(x) #全链接，激活函数
        x = F.log_softmax(x, dim =1) #log_softmax可以理解为概率对数值
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
original_net= torch.load('minst_conv_checkpoint')
print(original_net)

depth = [4, 8]

class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        # 两个并行的卷积通道，第一个通道：
        self.net1_conv1 = nn.Conv2d(1, 4, 5, padding=2)  # 一个输入通道，4个输出通道（4个卷积核），窗口为5，填充2
        self.net_pool = nn.MaxPool2d(2, 2)  # 2*2 池化
        self.net1_conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)  # 输入通道4，输出通道8（8个卷积核），窗口5，填充2

        # 第二个通道
        self.net2_conv1 = nn.Conv2d(1, 4, 5, padding=2)  # 一个输入通道，4个输出通道（4个卷积核），窗口为5，填充2
        self.net2_conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)  # 输入通道4，输出通道8（8个卷积核），窗口5，填充2

        # 全链接层
        self.fc1 = nn.Linear(2 * image_size // 4 * image_size // 4 * depth[1], 1024)  # 输入为处理后的特征图压平，输出1024个单元
        self.fc2 = nn.Linear(1024, 2 * num_classes)  # 输入1024个单元，输出20个单元
        self.fc3 = nn.Linear(2 * num_classes, num_classes)  # 输入20个单元，输出10个单元
        self.fc4 = nn.Linear(num_classes, 1)  # 输入10个单元，输出为1

    def forward(self, x, y, training=True):
        # 网络的前馈过程。输入两张手写图像x和y，输出一个数字表示两个数字的和
        # x,y都是batch_size*image_size*image_size形状的三阶张量
        # 输出为batch_size长的列向量
        # 首先，第一张图像进入第一个通道
        x = F.relu(self.net1_conv1(x))  # 第一层卷积
        x = self.net_pool(x)  # 第一层池化
        x = F.relu(self.net1_conv2(x))  # 第二层卷积
        x = self.net_pool(x)  # 第二层池化
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])  # 将特征图张量压平
        # 第2张图像进入第2个通道
        y = F.relu(self.net2_conv1(y))  # 第一层卷积
        y = self.net_pool(y)  # 第一层池化
        y = F.relu(self.net2_conv2(y))  # 第二层卷积
        y = self.net_pool(y)  # 第二层池化
        y = y.view(-1, image_size // 4 * image_size // 4 * depth[1])  # 将特征图张量压平

        # 将两个卷积过来的铺平向量拼接在一起，形成一个大向量
        z = torch.cat((x, y), 1)  # cat函数为拼接向量操作，1表示拼接的维度为第1个维度（0维度对应了batch）
        z = self.fc1(z)  # 第一层全链接
        z = F.relu(z)  # 对于深层网络来说，激活函数用relu效果会比较好
        z = F.dropout(z, training=self.training)  # 以默认为0.5的概率对这一层进行dropout操作
        z = self.fc2(z)  # 第二层全链接
        z = F.relu(z)
        z = self.fc3(z)  # 第三层全链接
        z = F.relu(z)
        z = self.fc4(z)  # 第四层全链接
        return z

    def set_filter_values(self, net):
        # 本函数为迁移网络所用，即将迁移过来的网络的权重值拷贝到本网络中
        # 本函数对应的迁移为预训练式
        # 输入参数net为从硬盘加载的网络作为迁移源

        self.net1_conv1.weight.data = copy.deepcopy(net.conv1.weight.data)
        self.net1_conv1.bias.data = copy.deepcopy(net.conv1.bias.data)
        self.net1_conv2.weight.data = copy.deepcopy(net.conv2.weight.data)
        self.net1_conv2.bias.data = copy.deepcopy(net.conv2.bias.data)
        self.net2_conv1.weight.data = copy.deepcopy(net.conv1.weight.data)
        self.net2_conv1.bias.data = copy.deepcopy(net.conv1.bias.data)
        self.net2_conv2.weight.data = copy.deepcopy(net.conv2.weight.data)
        self.net2_conv2.bias.data = copy.deepcopy(net.conv2.bias.data)

        # 将变量加载到GPU上
        self.net1_conv1 = self.net1_conv1.cuda() if use_cuda else self.net1_conv1
        self.net1_conv2 = self.net1_conv2.cuda() if use_cuda else self.net1_conv2

        self.net2_conv1 = self.net2_conv1.cuda() if use_cuda else self.net2_conv1
        self.net2_conv2 = self.net2_conv2.cuda() if use_cuda else self.net2_conv2

    def set_filter_values_nograd(self, net):
        # 本函数为迁移网络所用，即将迁移过来的网络的权重值拷贝到本网络中
        # 本函数对应的迁移为固定权重式
        # 调用set_filter_values为全部卷积核进行赋值
        self.set_filter_values(net)

        self.net1_conv1.weight.requires_grad = False
        self.net1_conv1.bias.requires_grad = False
        self.net1_conv2.weight.requires_grad = False
        self.net1_conv2.bias.requires_grad = False

        self.net2_conv1.weight.requires_grad = False
        self.net2_conv1.bias.requires_grad = False
        self.net2_conv2.weight.requires_grad = False
        self.net2_conv2.bias.requires_grad = False


# def rightness(predictions, labels):
#     """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行10列的矩阵，labels是数据之中的正确答案"""
#     pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
#     rights = pred.eq(labels.data.view_as(pred)).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
#     return rights, len(labels)  # 返回正确的数量和这一次一共比较了多少元素

def rightness(y, target):
    # 计算分类准确度的函数，y为模型预测的标签，target为数据的标签
    # 输入的y为一个矩阵，行对应了batch中的不同数据记录，列对应了不同的分类选择，数值对应了概率
    # 函数输出分别为预测与数据标签相等的个数，本次判断的所有数据个数
    out = torch.round(y.squeeze()).type(itype)
    out = out.eq(target).sum()
    out1 = y.size()[0]
    return(out, out1)

# 为了比较不同数据量对迁移学习的影响，设定了一个加载数据的比例fraction
# 即只加载原训练数据集的1/fraction来训练网络
fractions = [20, 10, 8, 6, 5, 4, 3, 2, 1]
# fractions = [50, 30]
with pd.ExcelWriter(FilePath) as writer:
    for fraction in fractions:
        # 生成网络实例
        net = Transfer().to(device)
        net.set_filter_values(original_net)
        if use_cuda:
            net = net.cuda()
        print(use_cuda)
        # 定义损失函数，我们用最小均方误差来定义损失
        criterion = nn.MSELoss()

        # 将需要训练的参数加载到优化器中
        new_parameters = []
        for para in net.parameters():
            if para.requires_grad:
                new_parameters.append(para)

        # 定义优化器
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
                # 代码其他处同样处理
                outputs = outputs.squeeze()
                labels = y1 + y2
                loss = criterion(outputs.float(), labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                right_train = rightness(outputs.float(), labels.float())
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
        ax1.set_ylim(0, 110)
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(df.index, df['Train_right'], label='Train_right', marker='.', linestyle='--', color='green')
        ax2.plot(df.index, df['Val_right'], label='Val_right', marker='*', linestyle='--', color='red')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='upper right')

        plt.title('TransferPretrained: fraction={} ; Test_right={:.2%}, Time_cost={:.2f}s'.format(fraction,test_right_ratio,time_cost))
        SaveFigPath=FigPath+'CNN_TransferPretrained_fraction_{}'.format(fraction)+'.png'
        plt.savefig(SaveFigPath,dpi=1200)

        plt.show()
        print('{}:写入保存完成'.format(SheetName))