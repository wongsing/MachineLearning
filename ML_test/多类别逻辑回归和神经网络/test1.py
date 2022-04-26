#实现手写数字0-9的识别，一对多分类，利用逻辑回归
#可以通过神经网络来实现，使用前馈神经网络进行预测
#将其当做是多个逻辑分类器

from cmath import cos
import imp
from random import sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.datasets import load_diabetes    #读取.mat文件也就是MATLAB格式文件
from sklearn.metrics import classification_report  #评估报告

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

#这部分内容还需学习，不太了解用法
#sharex/sharey 表示控制x、y轴之间的属性共享
#matshow(Mat,cmap=):绘制矩阵函数，,Mat为矩阵，cmap表示一种颜色映射方式
#xticks:设置刻度
fig,ax_array = plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(12,12))
for r in range(10):
    for c in range(10):
        ax_array[r,c].matshow(np.array(sample_images[10*r+c].reshape((20,20))).T,cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show() 

#sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#代价函数，带正则化
def cost(theta,X,y,learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    # reg = (learningRate)/2*len(X)*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    reg = theta.T.dot(theta) * learningRate/2*len(X)
    return np.sum(first-second)/len(X)+reg

#向量化梯度下降???为什么要转置
def gradient(theta,X,y,learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X*theta.T)-y

    grad = ((X.T * error) /len(X)).T + learningRate/len(X)*theta

    #正则化不惩罚0
    grad[0,0]=np.sum(np.multiply(error,X[:,0])) / len(X)
    return np.array(grad).ravel()

#一对多分类器--通过实现一对一全分类方法,k个不同类标签就有k个分类器，每个分类器在类别i 与 不是类别i 之间决定
#分类器训练包含在一个函数中，该函数计算10个分类器中的每个分类器的最终权重，并将权重返回为k*(n + 1)数组，其中n是参数数量。
def one_vs_all(X,y,num_labels,lamda):
    rows = X.shape[0]
    params = X.shape[1] #400个参数--》共需401个参数

    #k*(n+1)矩阵来作为k个分类器的参数
    all_theta = np.zeros((num_labels,params+1))

    #加一列常数项x0=1
    X = np.insert(X,0,values=np.ones(rows),axis=1)

    #训练标签为1~10，而0不分类，将类标签转化为二进制值（要么类i、要不不是类i）求出所有参数（参数数组）    
    for i in range(1,num_labels+1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i , (rows,1))

        fmin = minimize(fun=cost,x0=theta,args=(X,y_i,lamda),method='TNC',jac=gradient)
        all_theta[i-1,:] = fmin.x
    
    return all_theta

#向量化代码
rows = data['X'].shape[0]
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

#((5000, 401), (5000, 1), (401,), (10, 401))
print(X.shape, y_0.shape, theta.shape, all_theta.shape)
#查看y的标签数目
print(np.unique(data['y']))

all_theta = one_vs_all(data['X'],data['y'],10,1)
print(all_theta)



    

