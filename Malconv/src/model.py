import torch
import torch.nn as nn
import torch.nn.functional as F

# m  = nn.Conv1d(4, 128 , 500 , 500 )
# input= torch.randn(1 , 4 , 2000000)
# output = m(input)
# output.size()
# mm = nn.Sigmoid()
# weight = mm(output)
# x = weight * output
# mmm = nn.MaxPool1d(4000)
# x = mmm(x)
# x = x.view(-1, 128)
# mmmm = nn.Linear(128,128)
# mmmmm = nn.Linear(128,1)
# x = mmmm(x)
# x = mmmmm(x)

class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=500):
        super(MalConv, self).__init__()
        # 有257个词汇嵌入到8维向量空间中
        self.embed = nn.Embedding(257, 8, padding_idx=0)
        # 输入是4维输出128维
        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x)
        # 之所以要交换维度 是为了进行1维卷积的时候需要长度是字节的个数
        x = torch.transpose(x, -1, -2)
        # 取x的倒数第二维的，从索引0开始到 0 + 4 -1 范围的值
        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1, 128)
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x


