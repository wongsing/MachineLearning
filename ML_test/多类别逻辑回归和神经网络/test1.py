#实现手写数字0-9的识别，一对多分类，利用逻辑回归

import imp
from random import sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.datasets import load_diabetes    #读取.mat文件也就是MATLAB格式文件
from sklearn.metrics import classification_report #评估报告

# path = open('D:\Pytest\MachineLearning\ML_test\多类别逻辑回归和神经网络\ex3data1.mat')
data = loadmat("D:\Pytest\MachineLearning\ML_test\多类别逻辑回归和神经网络\ex3data1.mat")
print(data)
print(data['X'].shape,data['y'].shape)

#数据可视化
#随机显示100个数据
sample_idx = np.random.choice(np.arange(data['X'].shape[0]),100)
sample_images = data['X'][sample_idx,:]
print(data['X'].shape[0])
print(sample_idx)
print(sample_images)
print(sample_images.shape)

fig,ax_array = plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(12,12))
for r in range(10):
    for c in range(10):
        ax_array[r,c].matshow(np.array(sample_images[10*r+c].reshape((20,20))).T,cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()      

