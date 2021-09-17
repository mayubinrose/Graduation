import os
import time
import sys
import yaml
import numpy as np
import pandas as pd
from src.utils import ExeDataset,write_pred
from src.model import MalConv
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# argv[0] 是运行的文件名  1 是第一个参数 然后2 是第2个参数
try:
    config_path = sys.argv[1]
    seed = int(sys.argv[2])
    # 第一个是配置文件 对配置文件进行读取
    conf = yaml.load(open(config_path) , 'r')
except:
    print('Usage: python3 run_exp.py <config file path> <seed>')
    sys.exit()

# 实验名字
exp_name = conf['exp_name'] + '_sd_' + str(seed)
print("实验")
print('\t' , exp_name)

np.random.seed(seed)
torch.manual_seed(seed)


train_data_path = conf['train_data_path']
train_label_path = conf['train_label_path']

valid_data_path = conf['valid_data_path']
valid_label_path = conf['valid_label_path']

log_dir = conf['log_dir']
pred_dir = conf['pred_dir']
checkpoint_dir = conf['checkpoint_dir']


log_file_path = log_dir+exp_name+'.log'
chkpt_acc_path = checkpoint_dir+exp_name+'.model'
pred_path = pred_dir+exp_name+'.pred'

# 参数
