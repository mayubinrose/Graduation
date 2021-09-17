import csv
import pandas as pd
import os
import shutil
import string

md5_lables = []
lables = []
# 首先把所有的标签和md5的格式加入
with open('', 'r') as f:
    reader = csv.reader(f)
    all_lables = list(reader)
    for lable in all_lables:
        if lable[1][:9] != 'SINGLETON' and lable[1] != 'lable':
            md5_lables.append(lable)
            lables.append(lable[1])
family_name = []
# 标签看有多少种
counts = pd.value_counts(lables)
for key, value in counts.items():
    if value >= 200:
        family_name.append(key)


# 创建家族文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip()
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


if __name__ == '__main__':
    path = os.path.abspath('..')
    print(path)
    # 获取上一级的目录 创建文件夹
    for name in family_name:
        file_path = path + '\\' + 'Family' + '\\' + name
        print(file_path)
        mkdir(file_path)
    i = 0
    mal_path = '../data/VirusShare_00376'
    # 获取所有的文件和文件夹
    mal_list = os.listdir(mal_path)
    for md5_lable in md5_lables:
        if md5_lable[1] in family_name:
            i += 1
            dst_file_path = path + '\\' + 'Family' + '\\' + md5_lable[1] + '\\' + md5_lable[0]
            for mal_name in mal_list:
                if md5_lable[0] in mal_name:
                    src_file_path = mal_path + '\\' + mal_name
                    shutil.copy(src_file_path, dst_file_path)
