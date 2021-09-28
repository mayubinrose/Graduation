import torch
import time
import os
import sys
import yaml
import numpy as np
import pandas as pd
from src.utils import ExeDataset, write_pred
from src.model import MalConv
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 表示可以被检测到的显卡
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# 多GPU指定从0开始
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    config_path = sys.argv[1]
    seed = int(sys.argv[2])
    conf = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
except:
    print("运行出错！")
    sys.exit()

# 初始化神经网络的随机种子
np.random.seed(seed)
torch.manual_seed(seed)

# 参数
use_gpu = conf['use_gpu']
use_cpu = conf['use_cpu']
learning_rate = conf['learning_rate']
epoch_size = conf['epoch_size']
batch_size = conf['batch_size']
first_n_byte = conf['first_n_byte']
window_size = conf['window_size']
display_step = conf['display_step']
kfold = conf['kfold']

# 路径
train_valid_data_path = conf['train_data_path']
train_valid_label_path = conf['train_label_path']

test_data_path = conf['test_data_path']
test_label_path = conf['test_label_path']

log_dir = conf['log_dir']
pred_dir = conf['pred_dir']
checkpoint_dir = conf['checkpoint_dir']

# 实验名字
exp_name = conf['exp_name'] + '_sd_' + str(seed) + '_bs_' + str(batch_size) + '_ep_' + str(
    epoch_size)
train_log_file_path = log_dir + exp_name + '_train' + '.log'
valid_log_file_path = log_dir + exp_name + '_valid' + '.log'
test_log_file_path = log_dir + exp_name + '_test' + '.log'

chkpt_acc_path = checkpoint_dir + exp_name + '.model'
pred_path = pred_dir + exp_name + '_' + '.pred'
# 准备训练验证集的表格
tr_va_label_table = pd.read_csv(train_valid_label_path, header=None, index_col=0)
pd.set_option('display.max_rows', 2000)
tr_va_label_table = tr_va_label_table.rename(columns={1: 'ground_truth'})
tr_va_table = tr_va_label_table.groupby(level=0).last()
del tr_va_label_table
# print(tr_va_table)

# 准备测试集的表格
test_label_table = pd.read_csv(test_label_path, header=None, index_col=0)
test_label_table = test_label_table.rename(columns={1: 'ground_truth'})
test_table = test_label_table.groupby(level=0).last()
del test_label_table
# print(test_table)

print('Training and Valid Set:')
print('\tTotal', len(tr_va_table), 'files')
print('\tMalware Count :', tr_va_table['ground_truth'].value_counts()[1])
print('\tGoodware Count:', tr_va_table['ground_truth'].value_counts()[0])

print('Test Set:')
print('\tTotal', len(test_table), 'files')
print('\tMalware Count :', test_table['ground_truth'].value_counts()[1])
print('\tGoodware Count:', test_table['ground_truth'].value_counts()[0])

# tr_va_data 是加载数据之后按照batchsize分组，得到的就是tr_va_data的长度,默认不指定batch_size表示batch_size的长度为1
tr_va_data = ExeDataset(list(tr_va_table.index), train_valid_data_path, list(tr_va_table.ground_truth), first_n_byte)
# train_size 是34290 / batch_size * 0.8
train_size = (int)(0.8 * len(tr_va_data))
valid_size = len(tr_va_data) - train_size
train_set, valid_set = torch.utils.data.random_split(tr_va_data, [train_size, valid_size],
                                                     torch.Generator().manual_seed(seed))

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)
# 总体的测试集包含的元素个数
test_size = len(test_table)
testloader = DataLoader(ExeDataset(list(test_table.index), test_data_path, list(test_table.ground_truth), first_n_byte),
                        batch_size=batch_size, shuffle=True, num_workers=0)

# 准备一个test_idx 当做预测的时候存储在某一个预测文件内
test_idx = list(test_table.index)
print("epoch", epoch_size, "batch_size", batch_size)
print('训练数据总数', train_size, "训练数据的迭代次数", len(trainloader), "验证数据总数", valid_size, "验证数据的迭代次数", len(validloader),
      "测试数据总数", test_size, "测试数据的迭代次数", len(testloader))

malconv = MalConv(input_length=first_n_byte, window_size=window_size)
bce_loss = nn.BCEWithLogitsLoss()
adam_optim = optim.Adam([{'params': malconv.parameters()}], lr=learning_rate)
sigmoid = nn.Sigmoid()
if use_cpu:
    if torch.cuda.device_count() > 1:
        malconv = torch.nn.DataParallel(malconv)
    malconv = malconv.to(device)
    bce_loss = bce_loss.to(device)
    sigmoid = sigmoid.to(device)
# 打印在屏幕的
step_msg = '训练次数-{}-损失值-{:.6f}-acc-{:.4f}-单次训练时间-{:.2f} min'
valid_msg = '训练次数-{}-训练损失值-{:.6f}-训练准确率-{:.4f}-验证损失值-{:.6f}-验证准确率-{:.4f}'
# 打印在log上的
train_log = open(train_log_file_path, 'w', encoding="utf-8")
train_log.write("训练次数，训练损失值，训练准确率，单次训练时间/min\n")
train_log_msg = '{}, {:.6f}, {:.4f},{:.2f}'

valid_log = open(valid_log_file_path, 'w', encoding="utf-8")
valid_log.write('训练次数,训练损失值,训练准确率,验证损失值,验证准确率, 验证时间/min\n')
valid_log_msg = '{}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}'

test_log = open(test_log_file_path, 'w', encoding="utf-8")
test_log.write('epoch,测试损失值，测试准确率，测试时间/min\n')
test_log_msg = '{},{:.6f}, {:.4f}, {:.2f}'

history = {}
history['tr_loss'] = []
history['tr_acc'] = []

valid_best_acc = 0.0
total_step = 0
step_cost_time = 0

for i in range(epoch_size):
    print("第{}轮训练".format(i + 1))
    # enumerate 将一个元组或者列表变成一个索引数据
    for step, batch_data in enumerate(trainloader):
        # curbatch_size = batch_data[0].size(0) # 20
        # curbatch_size1 = batch_data[0].size() # 20 * 2000000
        # test = batch_data[1].size() # 20 * 1
        start = time.time()
        adam_optim.zero_grad()

        exe_input = batch_data[0].to(device) if use_gpu else batch_data[0]
        exe_intput = Variable(exe_input.long(), requires_grad=False)

        lable = batch_data[1].to(device) if use_gpu else batch_data[1]
        lable = Variable(lable.float(), requires_grad=False)

        pred = malconv(exe_input)
        loss = bce_loss(pred, lable)
        loss.backward()
        adam_optim.step()

        history['tr_loss'].append(loss.cpu().data.numpy())
        history['tr_acc'].extend(
            list(lable.cpu().data.numpy().astype(int) == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)))

        step_cost_time = time.time() - start

        total_step += 1
        # 每5次迭代输出一下
        if (step + 1) % display_step == 0:
            print(train_log_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
                                       step_cost_time / 60),
                  file=train_log, flush=True)
            print(step_msg.format(total_step, np.mean(history['tr_loss']),
                                  np.mean(history['tr_acc']), step_cost_time / 60, end='\r', flush=True))

    print("第{}轮训练结束，现在开始进行验证".format(i + 1))
    history['valid_loss'] = []
    history['valid_acc'] = []
    history['valid_pred'] = []
    validtime = time.time()
    malconv.eval()
    with torch.no_grad():
        for _, valid_batch_data in enumerate(validloader):
            exe_input = valid_batch_data[0].to(device) if use_gpu else valid_batch_data[0]
            exe_input = Variable(exe_input.long(), requires_grad=False)

            lable = valid_batch_data[1].to(device) if use_gpu else valid_batch_data[1]
            lable = Variable(lable.float(), requires_grad=False)

            pred = malconv(exe_input)
            loss = bce_loss(pred, lable)

            history['valid_loss'].append(loss.cpu().data.numpy())
            history['valid_acc'].extend(
                list(lable.cpu().data.numpy().astype(int) == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)))

            history['valid_pred'].append(list(sigmoid(pred).cpu().data.numpy()))
        print("第{}轮验证结束".format(i + 1), "现在开始输出log")
        print(valid_log_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
                                   np.mean(history['valid_loss']), np.mean(history['valid_acc']),
                                   (time.time() - validtime) / 60),
              file=valid_log, flush=True)
        print(valid_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
                               np.mean(history['valid_loss']), np.mean(history['valid_acc'])))
        if valid_best_acc < np.mean(history['valid_acc']):
            valid_best_acc = np.mean(history['valid_acc'])
            torch.save(malconv, chkpt_acc_path)
            print("模型保存在", chkpt_acc_path)
        history['tr_loss'] = []
        history['tr_acc'] = []

print("epoch={}轮训练全部结束，开始进行测试".format(epoch_size))
model = torch.load(chkpt_acc_path)
model = model.eval()
teststarttime = time.time()
history['test_loss'] = []
history['test_acc'] = []
with torch.no_grad():
    for _, test_batch_data in enumerate(testloader):
        exe_input = test_batch_data[0].to(device) if use_gpu else test_batch_data[0]
        exe_input = Variable(exe_input.long(), requires_grad=False)

        lable = test_batch_data[1].to(device) if use_gpu else test_batch_data[1]
        lable = Variable(lable.float(), requires_grad=False)

        pred = model(exe_input)
        loss = bce_loss(pred, lable)

        history['test_loss'].append(loss.cpu().data.numpy())
        history['test_acc'].extend(
            list(lable.cpu().data.numpy().astype(int) == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)))
    print(test_log_msg.format(epoch_size, np.mean(history['test_loss']), np.mean(history['test_acc']),
                              (time.time() - teststarttime) / 60),
          file=test_log, flush=True)
    print("测试结束，测试集的准确率为：{}".format(np.mean(history['test_acc'])))

train_log.close()
valid_log.close()
test_log.close()