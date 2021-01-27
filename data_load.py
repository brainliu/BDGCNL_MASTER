# _*_coding:utf-8_*_
# Name:Brian
# Create_time:2021/1/21 13:59
# file: data_load.py
# location:chengdu
# number:610000
import pandas as pd
import  numpy as np
import time
import copy
import torch
from torch.autograd import Variable
import torch
import matplotlib.pylab as plt
import torch.utils.data as utils

def construct_adj_matrix(distanc_path="./data/METR-LA/distances_la_2012.csv",
                         data_path='./data/METR-LA/metr-la.h5',
                         save_path="./data/METR-LA/la_2012_adj.npy"):
    distance_dict={}
    distance_data=pd.read_csv(distanc_path)
    for i in range(len(distance_data)):
        keys=str(distance_data.iloc[i,0])+str(distance_data.iloc[i,1])
        value_dis=distance_data.iloc[i,2]
        distance_dict[keys]=value_dis
    origin_Data=pd.read_hdf(data_path)
    sensors_id=origin_Data.columns
    adj_matrix=[]
    for id1 in sensors_id:
        adj_matx_temp=[]
        for id2 in sensors_id:
            keys_id12=str(id1)+str(id2)
            keys_id21=str(id2)+str(id1)
            temp=0
            if keys_id12==keys_id21:
                adj_matx_temp.append(1)
                continue
            if keys_id12 in distance_dict.keys():
                if distance_dict[keys_id12]:
                    adj_matx_temp.append(1)
                    continue
            if keys_id21 in distance_dict.keys():
                if distance_dict[keys_id21]:
                    adj_matx_temp.append(1)
                    continue
            else:
                adj_matx_temp.append(0)

        adj_matrix.append(adj_matx_temp)
    adj_result=np.array(adj_matrix)
    np.save(save_path,adj_result)
    return  adj_result




speed_matrix=pd.read_hdf('./data/PEMS-BAY/pems-bay.h5')

mask_ones_proportion=0.8 #不缺失的值的比率
seq_len=12
pred_len=12
shuffle=True
train_propotion=0.6
valid_propotion=0.2
BATCH_SIZE=32

def prepare_dataset(speed_matrix,BATCH_SIZE = 64,seq_len = 10, pred_len = 12,train_propotion = 0.7,
                    valid_propotion = 0.2, mask_ones_proportion = 0.8,random_seed = 1024):
    time_len = speed_matrix.shape[0]
    speed_matrix = speed_matrix.clip(0, 100)
    max_speed = speed_matrix.max().max()
    speed_matrix = speed_matrix / max_speed

    Mask = np.random.choice([0, 1], size=(speed_matrix.shape),
                                        p=[1 - mask_ones_proportion, mask_ones_proportion])
    masked_speed_matrix = np.multiply(speed_matrix, Mask)
    Mask[np.where(masked_speed_matrix == 0)] = 0 #将原本是0的也提取识别出来
    mask_zero_values = np.where(Mask == 0)[0].shape[0] / (Mask.shape[0] * Mask.shape[1])
    print('\t Masked dataset missing rate:', np.around(mask_zero_values, decimals=4), '(mask zero rate:',
          np.around(1 - mask_ones_proportion, decimals=4), ')')

    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(masked_speed_matrix.iloc[i:i + seq_len].values)
        speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)
    print('Input sequences, labels, and masks are generated.')
    # Mask sequences
    Mask = np.ones_like(speed_sequences)  #求最新的mask矩阵
    Mask[np.where(speed_sequences == 0)] = 0
    if shuffle: #随机打乱数据，但是连续的12个数据没有被打乱开来，如果要加时间标签，C就变成了2了，也就是12*2*352
        sample_size = speed_sequences.shape[0]
        index = np.arange(sample_size, dtype=int)
        np.random.shuffle(index)
        speed_sequences = speed_sequences[index]
        speed_labels = speed_labels[index]
        Mask = Mask[index]

    speed_sequences = np.expand_dims(speed_sequences, axis=1)
    Mask = np.expand_dims(Mask, axis=1)
    dataset_agger = np.concatenate((speed_sequences, Mask), axis = 1)
    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * ( train_propotion + valid_propotion)))
    train_data, train_label = dataset_agger[:train_index], speed_labels[:train_index]
    valid_data, valid_label = dataset_agger[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = dataset_agger[valid_index:], speed_labels[valid_index:]
    ##先不考虑融合时间信息和缺失值的情况
    ##但是我感觉需要加入一个特征，就是数据的时间标签，作为一个特征进去就行，先不考虑加吧。。
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    X_mean = np.mean(speed_sequences, axis=0)
    print('Finished')
    return  train_dataloader, valid_dataloader, test_dataloader, max_speed, X_mean

#计算邻接矩阵，前向和后向是不是不一样呢
def construct_adj_double(adj_data,steps=2):
    N = len(adj_data)
    adj = np.zeros([N * 2] * 2)
    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = adj_data

    for i in range(N):
        for k in range(steps-1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    for i in range(len(adj)):
        adj[i, i] = 1
    return adj

if __name__ == '__main__':

    construct_adj_matrix("./data/PEMS-BAY/distances_bay_2017.csv",
                             "./data/PEMS-BAY/pems-bay.h5",
                             "./data/PEMS-BAY/pems-bay_adj.npy")

    adj_data=np.load("./data/PEMS-BAY/pems-bay_adj.npy")
    adj_two=construct_adj_double(adj_data,steps=2)
    train_dataloader, valid_dataloader, test_dataloader, max_speed, X_mean=prepare_dataset(speed_matrix)




