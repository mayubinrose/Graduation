import os
import random
from random import choice
from shutil import copyfile
from shutil import copy
import time

ben_path = "/mnt/Data/myb/Grad/Data/benign/"
mal_path = "/mnt/Data/myb/Grad/Data/malware/"
train_path = "/mnt/Data/myb/Grad/Data/train/"
test_path = "/mnt/Data/myb/Grad/Data/test/"
seed = 2323


def createben_train_test(ben_path, train_path, test_path):
    for dirpath, dirnames, filenames in os.walk(ben_path):
        random.seed(seed)
        train_ben = random.sample(filenames, 18661)
        test_ben = []
        for i in filenames:
            if i not in train_ben:
                test_ben.append(i)
        print(len(train_ben))
        print(len(test_ben))
        for i in train_ben:
            srcfilename = os.path.join(dirpath, i)
            copy(srcfilename, train_path)
        for i in test_ben:
            srcfilename = os.path.join(dirpath, i)
            copy(srcfilename, test_path)


def createmal_train_test(mal_path, train_path, test_path):
    for dirpath, dirnames, filenames in os.walk(mal_path):
        print(len(filenames))
        train_mal = filenames[0:15629]
        test_mal = filenames[15629:]
        print(len(train_mal))
        print(len(test_mal))
        for i in train_mal:
            srcfilename = os.path.join(dirpath, i)
            copy(srcfilename, train_path)
        for i in test_mal:
            srcfilename = os.path.join(dirpath, i)
            copy(srcfilename, test_path)


if __name__ == '__main__':
    # 按照比例创建恶意的训练测试集合 训练测试比例 8：2 由于要构造一些对抗性的样本考虑到原始数据除去不含有PE结构的恶意代码之后的数量有限
    createmal_train_test(mal_path, train_path, test_path)
    # 按照比例创建良性的训练测试集合 训练测试比例 9：1
    createben_train_test(ben_path, train_path, test_path)

