# _*_coding:utf-8_*_
# Name:Brian
# Create_time:2021/2/3 13:40
# file: GCNM2.py
# location:chengdu
# number:610000
#设计一个GCNM模块，处理缺失值的填充，先按照单向的方式设计
#模型输入为一个时间点的xt 与xt+1，邻接矩阵，mt+1
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
device = "cuda"


class GCNM_BLOCK(nn.Module):
    #这里是单个的GCNM模块,输入为构建好的两个连续时间序列以及邻接矩阵，和mask
    def __init__(self,num_of_features):
        super(GCNM_BLOCK, self).__init__()
        self.FCconnected_GLU = nn.Linear(num_of_features, 2 * num_of_features).to(device)
    def forward(self,data,adj,mask_missing):
        """
        :param data: 两个连续时间的数据集data shape is (2N,B,C)
        :param adj: adj shape is (2N,2N)
        :param mask_missing: data shape is (N,N)表示缺失值的已知矩阵
        :return: shape is (N,B,C)
        """
        index=int(data.shape[0]/2)
        x_origin=data[index:,:,:]  #转化为 N B C 取后面的半个值
        data=torch.einsum('ii,ijk->ijk',adj,data)
        # print("GCN输出", data.shape)
        # shape is (2N, B, 2C') ,C'为filters的output维度
        data=self.FCconnected_GLU(data)
        lhs,rhs=torch.split(data,int(data.shape[2]/2),2)#shape is (2N, B, C'), (2N, B, C')
        #cropping 操作
        data=lhs*torch.sigmoid(rhs)
        data=data[index:,:,:] #这里为啥要取后面的值，不如取最大值

        # lhs,rhs=torch.split(data, int(data.shape[0] / 2), 0) #新加的可以取分裂以后的最大值
        # data = torch.max(torch.cat([lhs, rhs], 2), 2)[0].unsqueeze(-1)

        # print(data.shape)
        #缺失值的填充 N B C 返回值为这个
        # temp_result=data*(1-mask_missing)+x_origin*mask_missing
        # print(torch.isnan(temp_result).any())
        return data*(1-mask_missing)+x_origin*mask_missing

#在哪里构建这个2N的矩阵的问题，需要关注
class BDGCNM_BLOCK(nn.Module):
    def __init__(self,slide_length,num_of_features):
        super(BDGCNM_BLOCK,self).__init__()
        self.slide_length = slide_length #有多少个滑动窗口
        self.num_of_features=num_of_features
        self.muti_gcnm_forward = torch.nn.ModuleList().to(device)
        self.muti_gcnm_backward = torch.nn.ModuleList().to(device)
        for i in range(slide_length):
            self.muti_gcnm_forward.append(GCNM_BLOCK(num_of_features))
            self.muti_gcnm_backward.append(GCNM_BLOCK(num_of_features))
    def forward(self,data,adjfor,adjbac,mask_missing):
        """
        :param data: 长度为T的数据 q  N  B C
        :param mask_missing: 长度为q的mask_missing q N  B C
        :param adj:2N 2N
        :return: B q N C q个[N,B,C] q N B C
        """
        data=data.transpose(1,2) #预测区间的长度
        mask_missing=mask_missing.transpose(1,2)
        #计算h0的值
        data_impution_list=[[],[]] #用来保存计算结果，两个之间要融合一下
        fill_value=torch.mean(data).item() #增加了一个平均值填充的方法
        h0_for=torch.div(torch.sum(mask_missing,0),torch.sum(data,0)).to(device) # N B C
        h0_for = torch.where(torch.isnan(h0_for), torch.full_like(h0_for, fill_value), h0_for)
        h0_bac=torch.div(torch.sum(mask_missing,0),torch.sum(data,0)).to(device) #N B C
        h0_bac = torch.where(torch.isnan(h0_bac), torch.full_like(h0_bac, fill_value), h0_bac)

        result=[]
        #前向阶段
        for i in range(self.slide_length):
            # print(i)
            h_two=torch.cat([h0_for,data[i]],0) #2N B C
            mask_missing_t=mask_missing[i] #N B C
            h0_for=self.muti_gcnm_forward[i](h_two,adjfor,mask_missing_t) #N B C
            data_impution_list[0].append(h0_for)
        # del h0_for #消除内存占用
        # del h_two
        for j in range(self.slide_length-1,-1,-1):
            # print(j)
            h_two_back=torch.cat([data[j],h0_bac],0) #反过来拼接
            mask_missing_t = mask_missing[j]
            h0_bac = self.muti_gcnm_backward[j](h_two_back, adjbac, mask_missing_t)
            data_impution_list[1].append(h0_bac)
        # del h0_bac #消除内存占用
        # del h_two_back
        for k in range(self.slide_length):
        #取平均值,shape is  N  B C -> B N C -> B 1 N C
            temp=torch.mean(torch.stack([data_impution_list[0][k],data_impution_list[1][k] ]), 0).transpose(0,1)
            result.append(temp.unsqueeze(1))
        return torch.cat(result,1) # B q N C

#定义一个滑动的BDGCNM_layer
class SLIDE_bdgcnm_layer(nn.Module):
    def __init__(self,num_of_features,number_of_verticals,slide_length,seq_len):
        """
        :param num_of_features: 特征数量 C =1
        :param slide_length: 滑动窗口长度 q
        :param seq_len:  输入的序列长度 T
        """
        super(SLIDE_bdgcnm_layer, self).__init__()
        self.slide_length = slide_length
        self.num_of_features=num_of_features
        self.number_of_verticals=number_of_verticals
        self.seq_len = seq_len
        self.slide_box_num=seq_len-slide_length+1 #多少个滑动窗口数量用于捕获时间序列特征
        self.multi_bdgcn_blocks=torch.nn.ModuleList().to(device)
        self.multi_bdgcn_blocks2=torch.nn.ModuleList().to(device) #增加一层block来处理
        self.output_dims3=self.num_of_features*self.slide_length*self.slide_box_num
        for i in range(self.slide_box_num):
            #shape is B q N C,输出shape为这个
            self.multi_bdgcn_blocks.append(BDGCNM_BLOCK(slide_length,num_of_features))
            self.multi_bdgcn_blocks2.append(BDGCNM_BLOCK(slide_length,num_of_features))
    def forward(self,data,mask_missing,adjfor,adjbac):
        """
        :param data: T B  N C  输入长度为T的历史数据
        :param mask_missing: T B  N C missing标签
        : param adj: 邻接矩阵
        :return: B  N C*q*(T-q+1)=self.num_of_features*self.slide_length*self.slide_box_num
        """
        #先拆解数据，计算滑动窗口的数据集
        result_slide_box=[]
        for index in range(self.slide_box_num):
            data_temp=data[index:index+self.slide_length,:] # slide_length=q B N C
            mask_missing_temp=mask_missing[index:index+self.slide_length,:]
            #shape is B q N C
            #第一次silide box的填充结果
            # print(index)
            temp_slide_box1=self.multi_bdgcn_blocks[index](data_temp,adjfor,adjbac,mask_missing_temp)
            data_temp_2=temp_slide_box1.transpose(0,1) #转化为新的再进行一次处理
            temp_slide_box2 = self.multi_bdgcn_blocks2[index](data_temp_2, adjfor, adjbac, mask_missing_temp)
            #做了两次然后取最大值
            # temp=torch.mean(torch.stack([temp_slide_box1, temp_slide_box2 ]), 0)
            # result_slide_box.append(torch.mean(torch.stack([temp_slide_box1, temp_slide_box2 ]), 0))
            result_slide_box.append(torch.max(torch.cat([temp_slide_box1, temp_slide_box2], 3), 3)[0].unsqueeze(-1)) # slide_box_num 个 B q N C
        #shape B  N C*q*(T-q+1)
            # result_slide_box.append(temp_slide_box2)
        return torch.cat(result_slide_box,1).reshape(-1,self.number_of_verticals,self.output_dims3)

class BDGCNM_model(nn.Module):
    def __init__(self,number_of_verticals,num_of_features,slide_length,seq_len,predict_length,number_of_filters):
        super(BDGCNM_model,self).__init__()
        self.number_of_verticals=number_of_verticals
        self.num_of_features = num_of_features # 传感器数量N
        self.slide_length=slide_length #滑动窗口大小q
        self.seq_len = seq_len #输入的序列长度，T
        self.slide_box_num = seq_len - slide_length + 1  # 多少个滑动窗口数量用于捕获时间序列特征
        self.predict_length = predict_length
        self.number_of_filters=number_of_filters
        self.mask_for=nn.Parameter(torch.empty((2*number_of_verticals,2*number_of_verticals))).to(device)
        self.mask_bac=nn.Parameter(torch.empty((2*number_of_verticals,2*number_of_verticals))).to(device)
        self.input_dims = self.num_of_features * self.slide_length * self.slide_box_num
        self.SLIDE_bdgcnm_LAYER = SLIDE_bdgcnm_layer(num_of_features,number_of_verticals, slide_length, seq_len)
        self.outputlayer_1 = nn.ModuleList().to(device)
        self.output_layer2 = nn.ModuleList().to(device)
        self.output_layer3=nn.ModuleList().to(device)
        for i in range(predict_length):
            self.outputlayer_1.append(nn.Linear(self.input_dims,3*self.predict_length)).to(device)
            self.output_layer2.append(nn.Linear(3*self.predict_length, 2*self.predict_length)).to(device)
            self.output_layer3.append(nn.Linear(2 * self.predict_length, 1)).to(device)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.constant_(self.mask_for, 1.0)
        torch.nn.init.constant_(self.mask_bac, 1.0)
    def forward(self,data,adj,mask_missing):
        """
        :param data: T B  N C
        :param adj: 2N 2N
        :param mask_missing: T B  N C
        :return: B T' N
        """
        data = data.to(device)
        adj = adj.to(device)
        adj_for = self.mask_for*adj
        adj_bac=self.mask_bac*adj
        bdgcn_output=self.SLIDE_bdgcnm_LAYER(data,mask_missing,adj_for,adj_bac) #输出shape 是 B  N C*q*(T-q+1)
        result_predict=[]
        for j in range(self.predict_length):
            hidden=self.outputlayer_1[j](bdgcn_output)
            hidden = F.relu(hidden, inplace=True)
            hidden=self.output_layer2[j](hidden) #(B, 1, N)
            hidden = F.relu(hidden, inplace=True)
            hidden=self.output_layer3[j](hidden)
            result_predict.append(hidden.transpose(1,2))
        # del data
        # del hidden
        return torch.cat(result_predict,1) #(B, T', N)





