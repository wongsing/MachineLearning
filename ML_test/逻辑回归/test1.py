from cProfile import label
from inspect import Parameter
from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import accuracy_score

#训练一个逻辑回归模型预测，某个学生是否被大学录取
#训练集包括学生两次测试的评分，和最后的录取结果
#预测，通过两次测试的评分来决定是否被录取

#数据可视化
path = open('D:\Pytest\ML_test\逻辑回归\ex2data1.txt')
data = pd.read_csv(path,header=None,names=['Exam1','Exam2','Admitted'])
print(data.head())

positive = data[data['Admitted'].isin([1])]
# print(positive)
negative = data[data['Admitted'].isin([0])]

fig,ax = plt.subplots(figsize = (10,8))
ax.scatter(positive['Exam1'],positive['Exam2'],s=50,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam1'],negative['Exam2'],s=50,c='r',marker='x',label ='Not Admitted')
#legend：显示label图例
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

#sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#实现代价函数
def cost(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    return np.sum(first-second)/len(X)

#初始化X,y,theta
data.insert(0,'Ones',1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]       #从第二列到倒数第二列
y = data.iloc[:,cols-1:cols]    #取最后一列
theta = np.zeros(3)

#将X,y转换形式，从dataFrame 变成np，方便转成矩阵
X = np.array(X.values)
y = np.array(y.values)

#检查维度
# print(X.shape,y.shape,theta.shape)

#用初始化theta计算代价
print(cost(theta,X,y))

#梯度下降，没有进行更新，缺少学习速率和迭代次数
def gradient(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    #参数个数
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X*theta.T)-y

    for i in range(parameters):
        term = np.multiply(error,X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad

#用工具库计算theta值(不需自己定义迭代次数和学习速率)scipy.optimize.fmin_tnc
result = opt.fmin_tnc(func= cost , x0 = theta,fprime=gradient,args=(X,y))
# print(result),返回三个参数啊！

print(cost(result[0],X,y))

#画出决策曲线，x1代表30到100，‘y’是颜色
#自己选择的h(x)=g(@0+@1*x1+@2*x2)，所以才会有下面的式子
plotting_x1 = np.linspace(30,100,100)   
plotting_h1 = (-result[0][0]-result[0][1]*plotting_x1) / result[0][2]
print(result[0][0])
print(result[0][1])
print(result[0][2])
fig,ax = plt.subplots(figsize=(10,8))
ax.plot(plotting_x1,plotting_h1,'y',label='Prediction')
ax.scatter(positive['Exam1'],positive['Exam2'],s=50,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam1'],negative['Exam2'],s=50,c='r',marker='x',label ='Not Admitted')
#legend：显示label图例
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

#评价逻辑回归模型
def hfunc1(theta ,X ):
    return sigmoid(np.dot(theta.T,X))
print(hfunc1(result[0],[1,45,85]))


#另一种评价???
def predict(theta,X):
    probability = sigmoid(X*theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

print('result[0]=',result[0])
theta_min = np.matrix(result[0])
predictions = predict(theta_min,X)
correct = [1 if((a==1 and b==1) or (a==0 and b==0)) else 0 for (a,b) in zip(predictions,y)]
accuracy = (sum(map(int,correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))












