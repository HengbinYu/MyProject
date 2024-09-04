# 这个代码主要是利用ResNet18、ResNet34及ResNet50对MNIST数据集进行训练，选出最优的模型

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import pandas as pd
# 数据转换器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 数据加载器
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
print('DataLoader Complete')

class ResNetMNIST(nn.Module):
    def __init__(self, base_model):
        super(ResNetMNIST, self).__init__()
        self.resnet = base_model
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.resnet(x)


# 创建不同的ResNet模型实例
models = {
    'ResNet-18': ResNetMNIST(resnet18(weights=None)),
    'ResNet-34': ResNetMNIST(resnet34(weights=None)),
    'ResNet-50': ResNetMNIST(resnet50(weights=None)),
}
rate=0.5

def train_and_evaluate(model, trainloader, testloader, epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练和评估
    train_losses, test_losses, accuracies = [], [], []
    for epoch in range(epochs):
        model.train()
        # print('epoch: {}/{}'.format(epoch+1,epochs))
        running_loss = 0.0
        index=0
        train_num = math.ceil(len(trainloader) * rate)
        for inputs, labels in trainloader:
            if index>=train_num:
                break
            if index%50==0:
                print('epoch: {}/{},  Training:{}/{}'.format(epoch+1,epochs,index,train_num))
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            index+=1
        train_losses.append(running_loss / len(trainloader))
        print('train_losses:{}'.format(train_losses[-1]))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            test_loss = 0.0
            index = 0
            test_num = math.ceil(len(testloader) * rate)
            for inputs, labels in testloader:
                if index >= test_num:
                    break
                if index % 50 == 0:
                    print('epoch: {}/{},  Testing:{}/{}'.format(epoch+1,epochs,index, test_num))
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                index+=1
        test_losses.append(test_loss / len(testloader))
        accuracies.append(correct / total)
        print('test_losses:{}'.format(test_losses[-1]))
        print('accuracies:{}\n'.format(accuracies[-1]))
    return train_losses, test_losses, accuracies

print('start train All model')

results = {}
for name, model in models.items():
    print(f"Training {name}...\n\n")
    results[name] = train_and_evaluate(model, trainloader, testloader, epochs=10)
    print('\n',name,' Finished','\n\n')
print(type(results))
print(results)

def create_dataframe(data_dict, key):
    train_losses, test_losses, accuracies = data_dict[key]
    df = pd.DataFrame({
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracies': accuracies
    })
    return df

df_resnet18 = create_dataframe(results, 'ResNet-18')
df_resnet34 = create_dataframe(results, 'ResNet-34')
df_resnet50 = create_dataframe(results, 'ResNet-50')
with pd.ExcelWriter('resnets_result.xlsx') as writer:

    df_resnet18.to_excel(writer, sheet_name='ResNet-18')
    df_resnet34.to_excel(writer, sheet_name='ResNet-34')
    df_resnet50.to_excel(writer, sheet_name='ResNet-50')
print('Finished Write Results')
def plot_results1(results):
    plt.figure(figsize=(12, 8))
    for name, (train_losses, test_losses, accuracies) in results.items():
        plt.plot(accuracies, label=f'{name} Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Model Accuracy Comparison_{}.png'.format(rate), dpi=600)
    plt.close()
def plot_results2(results):
    plt.figure(figsize=(12, 8))
    for name, (train_losses, test_losses, accuracies) in results.items():
        plt.plot(train_losses, label=f'{name} train_losses')
    plt.title('Model Train_Losses Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Train_Losses')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Model Train_Losses Comparison_{}.png'.format(rate), dpi=600)
    plt.close()
def plot_results3(results):
    plt.figure(figsize=(12, 8))
    for name, (train_losses, test_losses, accuracies) in results.items():
        plt.plot(test_losses, label=f'{name} test_losses')
    plt.title('Model Test_Losses Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Test_Losses')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Model Test_Losses Comparison_{}.png'.format(rate), dpi=600)
    plt.close()
plot_results1(results)
plot_results2(results)
plot_results3(results)

# 选择并保存最佳模型
best_model = max(results, key=lambda x: max(results[x][2]))  # Select model with highest accuracy
torch.save(models['ResNet-18'].state_dict(), f'ResNet-18.pth')
torch.save(models['ResNet-34'].state_dict(), f'ResNet-34.pth')
torch.save(models['ResNet-50'].state_dict(), f'ResNet-50.pth')
torch.save(models[best_model].state_dict(), f'{best_model}_best.pth')
print(f"Saved {best_model} as the best model.")

