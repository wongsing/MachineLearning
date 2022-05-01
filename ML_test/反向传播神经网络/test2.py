from cmath import log
from random import uniform
from re import T
from unittest import result
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize

data = sio.loadmat('D:\Pytest\MachineLearning\ML_test\反向传播神经网络\ex4data1.mat')
raw_X = data['X']
raw_Y = data['y']

X = np.insert(raw_X,0,values=1,axis=1)
print(X.shape)

#对y进行独热编码处理： one-hot编码
def one_hot_encoder(raw_y):
    result= []
    for i in raw_y: #1-10
        y_temp = np.zeros(10)
        y_temp[i-1] = 1

        result.append(y_temp)
    
    return np.array(result)

y = one_hot_encoder(raw_Y)
print(y.shape,y[0])

#进行序列化权重参数-->由于进行到优化，传入的theta是一位数组 (n,)
theta = sio.loadmat('D:\Pytest\MachineLearning\ML_test\反向传播神经网络\ex4weights.mat')
theta1,theta2 = theta['Theta1'],theta['Theta2']
print(theta1.shape,theta2.shape)

def serialize(a,b):
    return np.append(a.flatten(),b.flatten())

theta_serialize = serialize(theta1,theta2)
print(theta_serialize.shape)

#解序列化权重参数
def deserialize(theta_serialize):
    theta1 = theta_serialize[:25*401].reshape(25,401)
    theta2 = theta_serialize[25*401:].reshape(10,26)
    return theta1,theta2

theta1,theta2=deserialize(theta_serialize)
print(theta1.shape,theta2.shape)

#正向传播
def sigmoid(z):
    return 1/(1+np.exp(-z))

def feed_forward(theta_serialize,X):
    theta1,theta2 = deserialize(theta_serialize)
    a1 = X
    z2 = a1 @ theta1.T  #@矩阵相乘==multiply 而*是矩阵的点乘 = A.dot(B) == np.dot(A,B)
    a2 = sigmoid(z2)
    a2 = np.insert(a2,0,values=1,axis=1)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1,z2,a2,z3,h

#不带正则化损失函数
def cost(theta_serialize,X,y):
    a1,z2,a2,z3,h = feed_forward(theta_serialize,X)
    J = -np.sum(y*np.log(h)+(1-y)*np.log(1-h))/len(X)
    return J

print(cost(theta_serialize,X,y))

#带正则化的损失函数
def reg_cost(theta_serialize,X,y,lamda):
    sum1 = np.sum(np.power(theta1[:,1:],2))
    sum2 = np.sum(np.power(theta2[:,1:],2))
    reg = (sum1+sum2)*lamda /(2*len(X))
    return reg + cost(theta_serialize,X,y)

lamda=1
print(reg_cost(theta_serialize,X,y,lamda))

#无正则化的梯度
def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def gradient(theta_serialize,X,y):
    theta1,theta2 = deserialize(theta_serialize)
    a1,z2,a2,z3,h = feed_forward(theta_serialize,X)
    d3 = h-y
    d2 = d3 @ theta2[:,1:]* sigmoid_gradient(z2)
    D2 = d3.T @ a2 / len(X)
    D1 = d2.T @ a1 / len(X)

    return serialize(D1,D2)

#带正则化参数
def reg_gradient(theta_serialize,X,y,lamda):
    D = gradient(theta_serialize,X,y)
    D1,D2 = deserialize(D)

    theta1,theta2 = deserialize(theta_serialize)
    D1[:,1:] = D1[:,1:] + theta1[:,1:]*lamda / len(X)
    D2[:,1:] = D2[:,1:] + theta2[:,1:]*lamda / len(X)

    return serialize(D1,D2)

#神经网络的优化
from scipy.optimize import minimize
def nn_training(X,y):
    #参数随机初始化
    init_theta = np.random.uniform(-0.5,0.5,10285)
    res = minimize(fun = reg_cost , x0 = init_theta,args=(X,y,lamda),method='TNC',jac=reg_gradient,options={'maxiter':300})
    return res

lamda = 1
res = nn_training(X,y)  #获取最优参数
raw_y = data['y'].reshape(5000,)  #转换成一维数组，要使用优化

_,_,_,_,h = feed_forward(res.x,X)

y_pred = np.argmax(h,axis=1)+1
acc = np.mean(y_pred == raw_y)

print('acc=',acc)

#不带正则化，acc=1 过拟合
#带正则化，lamda=10，acc = 0.934 / lamda = 1 0.9932

#隐藏层可视化，也就是隐藏层学到的特征
# def plot_hidden_layer(theta):
theta1,_ = deserialize(res.x)
hidden_layer = theta1[:,1:]
fig,ax = plt.subplots(nrows=5,ncols=5,figsize=(8,8),sharex=True,sharey=True)

for r in range(5):
    for c in range(5):
        ax[r,c].matshow(np.array(hidden_layer[5*r+c].reshape((20,20))),cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()


