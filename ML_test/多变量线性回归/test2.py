from inspect import Parameter
from os import path
import numpy as np
import pandas as pd


#多特征
path=open('D:\Pytest\ML_test\多变量线性回归\ex1data2.txt')
data2 = pd.read_csv(path,header= None,names=['Size','Bedrooms','Price'])
print(data2.head())

#特征归一化，常用的做法：每类特征-该特征的均值后再除以标准差，为的是减少迭代次数，使得梯度下降收敛更快
data2=(data2-data2.mean())/data2.std()
print(data2.head())

#代价函数
def computeCost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return  np.sum(inner) / (2*len(X))

#梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T)-y

        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j]-((alpha/len(X))*np.sum(term))
        
        theta = temp
        cost[i] = computeCost(X,y,theta)
    
    return theta,cost

data2.insert(0,'Ones',1)
cols = data2.shape[1]
X2=data2.iloc[:,0:cols-1]
y2=data2.iloc[:,cols-1:cols]

X2=np.matrix(X2.values)
y2=np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

#初始化参数，学习速率a和迭代次数
alpha = 0.01
iters = 1500


g2,cost2 = gradientDescent(X2,y2,theta2,alpha,iters)
print(g2)

#正规方程
def normalEqn(X,y):
    theta=np.linalg.inv(X.T@X)@X.T@y
    return theta

final_theta2 = normalEqn(X2,y2)
print(final_theta2)

#data1
path=open('D:\Pytest\ML_test\单变量的线性回归\\t.txt')
data1 = pd.read_csv(path,header=None,names=['Population','Profit'])
data1.insert(0,'Ones',1)
cols=data1.shape[1] 
X1 = data1.iloc[:,:-1]    #iloc[:,:] 前面是行，后面是列，倒数二列
y1=data1.iloc[:,cols-1:cols]  #y是data最后一列，利润

final_theta1 = normalEqn(X1,y1)
print(final_theta1)