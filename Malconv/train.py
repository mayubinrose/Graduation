import os
import time
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# argv[0] 是运行的文件名  1 是第一个参数 然后2 是第2个参数
try:
    config_path = sys.argv[1]
    seed = int(sys.argv[2])
    # 第一个是配置文件 对配置文件进行读取
    conf = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
except:
    print('Usage: python3 run_exp.py <config file path> <seed>')
    sys.exit()

# 获取一个随机种子
np.random.seed(seed)
torch.manual_seed(seed)

# 参数
use_gpu = conf['use_gpu']
use_cpu = conf['use_cpu']
learning_rate = conf['learning_rate']
epoch_size = conf['epoch_size']
test_step = conf['test_step']
batch_size = conf['batch_size']
first_n_byte = conf['first_n_byte']
window_size = conf['window_size']
display_step = conf['display_step']
sample_cnt = conf['sample_cnt']
# 路径
train_data_path = conf['train_data_path']
train_label_path = conf['train_label_path']

test_data_path = conf['test_data_path']
test_label_path = conf['test_label_path']

log_dir = conf['log_dir']
pred_dir = conf['pred_dir']
checkpoint_dir = conf['checkpoint_dir']

# 实验名字
exp_name = conf['exp_name'] + '_sd_' + str(seed) + str(batch_size)
log_file_path = log_dir + exp_name + '.log'
chkpt_acc_path = checkpoint_dir + exp_name + '.model'
pred_path = pred_dir + exp_name + '_' + '.pred'

# header =0  表示我们将第一行视为标题 如果header = None 表示我们不将第一行视作标题 而是看作数据的一部分
tr_label_table = pd.read_csv(train_label_path, header=None,
                             index_col=0)  # 默认情况下读取将添加一个列index，如果我们不想添加那么指定采用哪一列作为index_col index_col指出哪一列作为索引
# 设置最大的打印行
pd.set_option('display.max_rows', 20000)

tr_label_table.index = tr_label_table.index
# 将列名重新命名
tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})

test_label_table = pd.read_csv(test_label_path, header=None, index_col=0)
# test_label_table.index = test_label_table.index.str.upper()
test_label_table.index = test_label_table.index
test_label_table = test_label_table.rename(columns={1: 'ground_truth'})

# 合并表格并移除重复项 指定第一个索引，level = 0 表示第一个索引
tr_table = tr_label_table.groupby(level=0).last()
del tr_label_table
test_table = test_label_table.groupby(level=0).last()
del test_label_table
# print(tr_table)
print('Training Set:')
print('\tTotal', len(tr_table), 'files')
print('\tMalware Count :', tr_table['ground_truth'].value_counts()[1])
print('\tGoodware Count:', tr_table['ground_truth'].value_counts()[0])

print('Test Set:')
print('\tTotal', len(test_table), 'files')
print('\tMalware Count :', test_table['ground_truth'].value_counts()[1])
print('\tGoodware Count:', test_table['ground_truth'].value_counts()[0])

# 如果不是1 那么表示每次要随机取多少次样本
if sample_cnt != 1:
    tr_table = tr_table.sample(n=sample_cnt, random_state=seed)

dataloader = DataLoader(ExeDataset(list(tr_table.index), train_data_path, list(tr_table.ground_truth), first_n_byte),
                        batch_size=batch_size, shuffle=True, num_workers=use_gpu)

testloader = DataLoader(ExeDataset(list(test_table.index), test_data_path, list(test_table.ground_truth), first_n_byte),
                        batch_size=batch_size, shuffle=True, num_workers=use_gpu)

test_idx = list(test_table.index)

del tr_table
del test_table

malconv = MalConv(input_length=first_n_byte, window_size=window_size)
bce_loss = nn.BCEWithLogitsLoss()
adam_optim = optim.Adam([{'params': malconv.parameters()}], lr=learning_rate)
sigmoid = nn.Sigmoid()

if use_gpu:
    if torch.cuda.device_count() > 1:
        malconv = torch.nn.DataParallel(malconv)
    # malconv.cuda()
    malconv = malconv.to(device)
    # bce_loss = bce_loss.cuda()
    bce_loss = bce_loss.to(device)
    # sigmoid = sigmoid.cuda()
    sigmoid = sigmoid.to(device)

step_msg = 'step-{}-loss-{:.6f}-acc-{:.4f}-time-{:.2f}'
test_msg = 'step-{}-tr_loss-{:.6f}-tr_acc-{:.4f}-test_loss-{:.6f}-test_acc-{:.4f}'
log_msg = '{}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}'
history = {}
history['tr_loss'] = []
history['tr_acc'] = []

log = open(log_file_path, 'w')
log.write('step,tr_loss , tr_acc , test_loss , test_acc,time\n')

test_best_acc = 0.0
total_step = 0
step_cost_time = 0

#
while total_step < 1000:
    # 训练开始
    # enumerate 将一个元组或者列表变成一个索引数据
    for step, batch_data in enumerate(dataloader):
        start = time.time()

        adam_optim.zero_grad()
        cur_batch_size = batch_data[0].size(0)

        # exe_input = batch_data[0].cuda() if use_gpu else batch_data[0]
        exe_input = batch_data[0].to(device) if use_gpu else batch_data[0]
        exe_input = Variable(exe_input.long(), requires_grad=False)

        # lable = batch_data[1].cuda() if use_gpu else batch_data[1]
        lable = batch_data[1].to(device) if use_gpu else batch_data[1]
        lable = Variable(lable.float(), requires_grad=False)

        pred = malconv(exe_input)
        # print(pred)
        loss = bce_loss(pred, lable)
        # print(loss)
        loss.backward()
        adam_optim.step()

        # append只是简单的添加一个列表 extend是将列表中的元素一个个添加进去
        # history['tr_loss'].append(loss.cpu().data.numpy()[0])
        history['tr_loss'].append(loss.cpu().data.numpy())
        history['tr_acc'].extend(
            list(lable.cpu().data.numpy().astype(int) == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)))

        step_cost_time = time.time() - start

        if (step + 1) % display_step == 0:
            print(step_msg.format(total_step, np.mean(history['tr_loss']),
                                  np.mean(history['tr_acc']), step_cost_time, end='\r', flush=True))
        total_step += 1
        # 中断然后进行测试
        if total_step % test_step == 0:
            break

    # 测试阶段
    history['test_loss'] = []
    history['test_acc'] = []
    history['test_pred'] = []

    for _, test_batch_data in enumerate(testloader):
        cur_batch_size = test_batch_data[0].size(0)

        # exe_input = test_batch_data[0].cuda() if use_gpu else test_batch_data[0]
        exe_input = test_batch_data[0].to(device) if use_gpu else test_batch_data[0]
        exe_input = Variable(exe_input.long(), requires_grad=False)

        # lable = test_batch_data[1].cuda() if use_gpu else test_batch_data[1]
        lable = test_batch_data[1].to(device) if use_gpu else test_batch_data[1]
        lable = Variable(lable.float(), requires_grad=False)

        pred = malconv(exe_input)
        loss = bce_loss(pred, lable)

        history['test_loss'].append(loss.cpu().data.numpy())
        history['test_acc'].extend(
            list(lable.cpu().data.numpy().astype(int) == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)))
        history['test_pred'].append(list(sigmoid(pred).cpu().data.numpy()))

    print(log_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
                         np.mean(history['test_loss']), np.mean(history['test_acc']), step_cost_time),
          file=log, flush=True)
    print(test_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
                          np.mean(history['test_loss']), np.mean(history['test_acc'])))
    if test_best_acc < np.mean(history['test_acc']):
        test_best_acc = np.mean(history['test_acc'])
        torch.save(malconv, chkpt_acc_path)
        print('Checkpoint saved at', chkpt_acc_path)
        #
        write_pred(history['test_pred'], test_idx, pred_path)
        print('Prediction saved at', pred_path)

    history['tr_loss'] = []
    history['tr_acc'] = []
