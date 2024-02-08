import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义卷积层、池化层和全连接层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        # 定义前向传播
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 实例化模型、定义损失函数和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# 训练模型
num_epochs = 30
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 用于累积每个 epoch 的训练损失
    for inputs, labels in tqdm(trainloader):  # 迭代训练数据集
        inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到设备上
        optimizer.zero_grad()  # 梯度清零，防止梯度累积
        outputs = model(inputs)  # 前向传播，计算模型输出
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        running_loss += loss.item()  # 累积当前 batch 的损失
    # 每个 epoch 结束后进行学习率调度
    scheduler.step()
    # 打印训练损失
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(trainloader)}')
# 在测试集上评估模型
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
accuracy = correct / total
print(f'Test Accuracy: {accuracy}')
# 计算每个类别的 precision、recall 和 F1-score
precision_per_class = precision_score(all_labels, all_preds, average=None)
recall_per_class = recall_score(all_labels, all_preds, average=None)
f1_per_class = f1_score(all_labels, all_preds, average=None)
# 计算平均值
avg_precision = precision_score(all_labels, all_preds, average='macro')
avg_recall = recall_score(all_labels, all_preds, average='macro')
avg_f1 = f1_score(all_labels, all_preds, average='macro')
# 输出每个类别的 precision、recall 和 F1-score
for i in range(10):
    print(f'Class {i} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1-score: {f1_per_class[i]:.4f}')
# 输出平均值
print(f'Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1-score: {avg_f1}')
# 输出更详细的分类报告
report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)])
print(report)
