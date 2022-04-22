from cProfile import label
from distutils.log import error
from inspect import Parameter
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#生成主对角线全1，其余全0的矩阵
A = np.eye(5)
print(A)

#单变量的线性回归
#根据城市人口的数量，预测开小吃店的利润！

#首先，读入数据，展示数据
#必须先通过open，如果文件名有中文的话;注意t要转义\
path=open('D:\Pytest\ML_test\单变量的线性回归\\t.txt')
data = pd.read_csv(path,header=None,names=['Population','Profit'])
print(data.head())

#pyplot 中的 scatter() 方法来绘制散点图,figsize指的是图标的大小，长宽
data.plot(kind='scatter',x='Population',y='Profit',figsize=(10,8))
plt.show()

#梯度下降+代价函数，在数据集上，训练线性回归的参数theta
#由于乘法是点乘，则增加一列为1，乘的时候就是1*@0 + Population(x)*@1
#代价函数公式: J(0)=(1/2m) *[累加][h(x)-y]^2
#假设函数: h(x)=@0+@1*x(线性回归)
def computeCost(X,y,theta):
    inner=np.power(((X*theta.T-y)),2)
    return np.sum(inner)/(2*len(X))

#实现，先加一列x用来更新参数值，先将参数值设为0，学习率为0.01，迭代次数为1500次
#假设函数h(x)=@0+@1*x，由于乘法是点乘，则增加一列为1，乘的时候就是1*@0 + Population(x)*@1
data.insert(0,'Ones',1)
cols=data.shape[1] 
X = data.iloc[:,:-1]    #iloc[:,:] 前面是行，后面是列，倒数二列
y=data.iloc[:,cols-1:cols]  #y是data最后一列，利润
print(cols)
print(X.to_string())
print(y.to_string())

#代价函数为numpy矩阵，需要转换X/y，并且初始化参数
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))

#查看维度
print('X的维度：',X.shape,'y的维度：',y.shape,'theta的维度：',theta.shape)

#计算代价函数J,结果是32.07
print(computeCost(X,y,theta))

#梯度下降，通过变化theta来使得J函数最小，迭代逐步接近代价函数最小,学习速率a和迭代次数
#正常工作流程：打印出每一步的J的值，看是否一直在减少，直到最后收敛到一个稳定值
#梯度下降，就是求两个参数
def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    #进行计算，j=1时，1/𝑚 ∑ ((ℎ𝜃(𝑥(𝑖)) − 𝑦(𝑖)) ⋅ 𝑥(𝑖))
    for i in range(iters):
        error = (X*theta.T)-y

        for j in range(parameters):
            term = np.multiply(error,X[:,j])    
            temp[0,j] = theta[0,j]-((alpha/len(X)) * np.sum(term))  #参数

        theta = temp
        cost[i] = computeCost(X,y,theta)
    
    return theta,cost

#初始化参数，学习速率a和迭代次数
alpha = 0.01
iters = 1500

#开始运行梯度下降算法，将参数适用于训练集
g,cost=gradientDescent(X,y,theta,alpha,iters)
print(g)

#预测35000和70000城市规模的小吃摊利润
predict1 = [1,3.5]*g.T
print('predict1:',predict1)
predict2 = [1,7]*g.T
print('predict2:',predict2)

#输出原始数据和拟合的直线
x=np.linspace(data.Population.min(),data.Population.max(),100)
f =g[0,0]+(g[0,1]*x)

fig,ax=plt.subplots(figsize=(10,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
