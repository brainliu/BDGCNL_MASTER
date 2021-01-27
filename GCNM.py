# _*_coding:utf-8_*_
# Name:Brian
# Create_time:2021/1/27 15:56
# file: GCNM.py
# location:chengdu
# number:610000
#设计一个GCNM模块，处理缺失值的填充，先按照单向的方式设计
#模型输入为一个时间点的xt 与xt+1，邻接矩阵，mt+1
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda")
class GCN_layer(nn.Module):
    def __init__(self,num_of_filter,num_of_features):
        super(GCN_layer, self).__init__()
        self.FCconnected_GLU = nn.Linear(num_of_features, 2 * num_of_filter).to(device)
    def forward(self,data,adj):
        ##这里面的
        # print("GCN输入",data.shape, adj.shape)
        #data shape is (3N,B,C) ,adj shape is (3N,3N)
        data=torch.einsum('ii,ijk->ijk',adj,data)
        # print("GCN输出", data.shape)
        # shape is (3N, B, 2C') ,C'为filters的output维度
        data=self.FCconnected_GLU(data)
        lhs,rhs=torch.split(data,int(data.shape[2]/2),2)#shape is (3N, B, C'), (3N, B, C')
        del data
        # print("GLU输出shape",lhs.shape,rhs.shape)
        return lhs*torch.sigmoid(rhs)
