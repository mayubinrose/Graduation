import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
import time
from model import *
import os
# 表示可以被检测到的显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 多GPU指定从0开始
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

train_data = torchvision.datasets.CIFAR10(root='../data', train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)

test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

# 利用DataLoader来加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)



mymodel = Mymodel()

print((mymodel))
if torch.cuda.device_count() > 1:
    # 两种方法 要么在这里声明要么直接os.enriron
    #mymodel = torch.nn.DataParallel(mymodel,device_ids=[0,1])
    mymodel = torch.nn.DataParallel(mymodel)
mymodel = mymodel.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(mymodel.parameters(), learning_rate)

# 設置網絡參數
# 記錄訓練的次數
total_train_step = 0
total_test_step = 0

epoch = 10
writer = SummaryWriter("../logs_train")
start_time = time.time()

for i in range(epoch):
    print("------------第{}轮训练开始----------".format(i + 1))

    # 训练步骤开始
    mymodel.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = mymodel(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss:{}".format(total_train_step, loss.item()))
            # item函数显示为数值而不是tensor类型的
            writer.add_scalar("train_loss" , loss.item() , total_train_step)

    mymodel.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs , targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = mymodel(imgs)
            loss = loss_fn(outputs , targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(mymodel.state_dict() , "../model/mymodel_{}.pth".format(i))
    print("模型已保存")
writer.close()