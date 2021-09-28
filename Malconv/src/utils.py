import numpy as np
import torch
from torch.utils.data import Dataset

# write_pred(history['test_pred'], test_idx, pred_path)
# 将测试集的预测结果值 保存在一个文件夹下，test_idx 是当前的文件名，文件名加上预测值存储在file_path下面
def write_pred(test_pred, test_idx, file_path):
    # 在test_pred下的sublist 下的item 一个个添加到test_pred list下面
    test_pred = [item for sublist in test_pred for item in sublist]
    # 写下预测值 zip函数的方法是将对象中对应的元素打包成一个个元组，然后返回这些元组组成的列表
    with open(file_path, 'w') as f:
        for idx, pred in zip(test_idx, test_pred):
            print(idx.upper() + ',' + str(pred[0]), file=f)

# 数据集的预处理
class ExeDataset(Dataset):
    # data_path是路径 lable_list 是标签列表 fp_list 是文件列表
    def __init__(self, fp_list, data_path, lable_list, first_n_byte=2000000):
        self.fp_list = fp_list
        self.data_path = data_path
        self.lable_list = lable_list
        self.first_n_byte = first_n_byte
    # 得到当前的文件的长度
    def __len__(self):
        return len(self.fp_list)
    # 将文件补充到2000000 个字节的长度之后，然后返回array形式的
    def __getitem__(self, idx):
        try:
            # 读取前200 0000 个字节 这些字节
            with open(self.data_path + self.fp_list[idx], 'rb') as f:
                tmp = [i + 1 for i in f.read()][:self.first_n_byte]
                tmp = tmp + [0] * (self.first_n_byte - len(tmp))
        except:
            with open(self.data_path + self.fp_list[idx], 'rb') as f:
                tmp = [i + 1 for i in f.read()][:self.first_n_byte]
                tmp = tmp + [0] * (self.first_n_byte - len(tmp))
        return np.array(tmp), np.array([self.lable_list[idx]])
