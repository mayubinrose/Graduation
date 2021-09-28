import os
#import pandas as pd
import csv
train_path  = "/mnt/Data/myb/Grad/Graduation/Malconv/data/train/"
test_path = "/mnt/Data/myb/Grad/Graduation/Malconv/data/test/"
train_lable_path = "/mnt/Data/myb/Grad/Graduation/Malconv/data/train_lable.csv"
test_lable_path = "/mnt/Data/myb/Grad/Graduation/Malconv/data/test_lable.csv"

def create_csv(lable_path , train_path):
    with open(lable_path , 'w') as f:
        csv_write = csv.writer(f)
        for dirpath , dirnames ,filenames in os.walk(train_path):
            for filename in filenames:
                row = []
                # 良性的标签为0
                if "exe" in filename:
                    row = [filename , "0"]
                else:
                    row = [filename , "1"]
                csv_write.writerow(row)


if __name__ == '__main__':
    # 创建训练集的csv
    create_csv(train_lable_path, train_path)
    # 创建测试机的csv
    create_csv(test_lable_path, test_path)