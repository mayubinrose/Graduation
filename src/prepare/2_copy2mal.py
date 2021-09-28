import csv
import pandas as pd
import os
import shutil
import re
import sys
# 第一步调用virustotal中的api得到json文件，第二步根据json文件调用avclass函数得到txt文件，然后我们自行转换为csv文件，然后开始进入这一步收集到各个地方的含有PE结构的恶意代码
mal_csv = r'/mnt/Data/myb/Grad/Graduation/output.csv'
dst_mal_path = '/mnt/Data/myb/Grad/Data/malware/'
src_mal_path = "/mnt/Data/myb/Grad/Graduation/Grade1/data/VirusShare_00376"
family_path = '/mnt/Data/myb/研一实验/Family/'
testfamily_path = '/mnt/Data/myb/研一实验/TestFamily/'
example_path  =  "/mnt/Data/myb/毕业师兄姐课题/吴睿-交接/malware_new/examples/"
example1_path = "/mnt/Data/myb/毕业师兄姐课题/吴睿-交接/malware_new/examples1/"
oriad_mal_path = "/mnt/Data/myb/Grad/Data/oriad_malware/"

def copy2mal(mal_csv, src_mal_path, dst_mal_path):
    md5s = []
    with open(mal_csv, 'r') as f:
        reader = csv.reader(f)
        for item in reader:
            if item[1][0:9] != 'SINGLETON':
                md5s.append(item[0])
    i = 0
    for md5 in md5s:
        md5_2 = "VirusShare_" + md5
        srcpath = os.path.join(src_mal_path, md5_2)
        size = os.path.getsize(srcpath)
        if size / 1024 < 1:
            continue

        if not os.path.isfile(srcpath):
            print(md5_2 + 'is not file')

        with open(srcpath, 'rb') as fp:
            flag1 = fp.read(2)
            fp.seek(0x3c)
            offset = ord(fp.read(1))
            fp.seek(offset)
            flag2 = fp.read(4)
            if flag1 == b'MZ' and flag2 == b'PE\x00\x00':
                print(md5_2 + ' is a PE file')
                dstpath = os.path.join(dst_mal_path, md5)
                shutil.copy(srcpath, dstpath)
                i += 1
                print(i)

def copyfamily2mal(family_path ,dst_mal_path ):
    i = 0
    for dirpath , dirname , filenames in os.walk(family_path):
        for file in  filenames:
            srcpath = os.path.join(dirpath , file)
            size = os.path.getsize(srcpath)
            if size / 1024 < 1:
                continue

            if not os.path.isfile(srcpath):
                print(srcpath + 'is not file')

            with open(srcpath , 'rb') as fp:
                flag1 = fp.read(2)
                fp.seek(0x3c)
                offset = ord(fp.read(1))
                fp.seek(offset)
                flag2 = fp.read(4)
                if flag1 == b'MZ' and flag2==b'PE\x00\x00':
                    print(srcpath+ ' is a PE file')
                    if 'virusshare_' in file:
                        file = file[11:]
                    dstpath =os.path.join(dst_mal_path , file)
                    print(dstpath)
                    shutil.copy(srcpath , dstpath)
                    i += 1
                    print(i)




if __name__ == '__main__':
    # 将所有的含有PE文件的VirusShare_00376 转移到malware文件夹下 去掉前缀VirusShare_
    copy2mal(mal_csv ,src_mal_path ,  dst_mal_path)
    # 将family中的文件转移到malware文件夹中
    # copyfamily2mal(family_path ,  dst_mal_path)
    # 将TestFamily 中的文件转移到malware中
    # copyfamily2mal(testfamily_path ,  dst_mal_path)




