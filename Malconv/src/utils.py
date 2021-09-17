import numpy as np
import torch
from torch.utils.data import Dataset


def write_pred(test_pred, test_idx, file_path):
    test_pred = [item for sublist in test_pred for item in sublist]
    with open(file_path, 'w') as f:
        for idx, pred in zip(test_idx, test_pred):
            print(idx.upper() + ',' + str(pred[0]), file=f)


class ExeDataset(Dataset):
    # data_path是路径 lable_list 是标签列表 fp_list 是文件列表
    def __init__(self, fp_list, data_path, lable_list, first_n_byte=2000000):
        self.fp_list = fp_list
        self.data_path = data_path
        self.lable_list = lable_list
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        try:
            with open(self.data_path + self.fp_list[idx], 'rb') as f:
                tmp = [i + 1 for i in f.read()][:self.first_n_byte]
                tmp = tmp + [0] * (self.first_n_byte - len(tmp))
        except:
            with open(self.data_path + self.fp_list[idx].lower(), 'rb') as f:
                tmp = [i + 1 for i in f.read()][:self.first_n_byte]
                tmp = tmp + [0] * (self.first_n_byte - len(tmp))
        return np.array(tmp), np.array([self.lable_list[idx]])
