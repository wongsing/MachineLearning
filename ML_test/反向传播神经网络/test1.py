#使用反向传播的前馈神经网络实现手写数字集的一个识别，自动学习神经网络中的参数
from json import encoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

#数据可视化
data = loadmat('D:\Pytest\MachineLearning\ML_test\反向传播神经网络\ex4data1.mat')
print(data)

X = data['X']
y = data['y']
print('X:',X.shape,',y:',y.shape)
print('y[1,:]:',y[1,:])
#X(5000,400),y(5000,1)
weight = loadmat('D:\Pytest\MachineLearning\ML_test\反向传播神经网络\ex4weights.mat')
theta1,theta2 = weight['Theta1'],weight['Theta2']
print('theta1:',theta1.shape,',theta2:',theta2.shape)
# print(theta1)
# print('theta1[:,1:]=',theta1[:,1:])

sample_idx = np.random.choice(np.arange(data['X'].shape[0]),100)
sample_images = data['X'][sample_idx,:]
fig,ax_array = plt.subplots(nrows = 10,ncols=10,sharey=True,sharex=True,figsize=(12,12))
for r in range(10):
    for c in range(10):
        ax_array[r,c].matshow(np.array(sample_images[10*r+c].reshape((20,20))).T,cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()

#前向传播和代价函数

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_propagate(X,theta1,theta2):
    m = X.shape[0]  #X.shape = (5000,400)，所以一列要加5000个1
    a1 = np.insert(X,0,values=np.ones(m),axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2),0,values=np.ones(m),axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1,z2,a2,z3,h

#不带正则化的代价函数,输入层、隐藏层，类标签，学习速率感觉没啥用啊？
def cost(theta1,theta2,input_size,hidden_size,num_labels,X,y,learning_rate):
    m = X.shape[0]  #m=5000
    X = np.matrix(X)
    y = np.matrix(y)

    #计算前向传播
    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)

    #计算代价函数,y与h（假设函数）的预测值一一对应
    J = 0
    for i in range(m):
        first_term = np.multiply((-y[i,:]),np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J += np.sum(first_term-second_term)
    
    J = J/m
    return J

#对y进行编码，将5000*1的向量变成5000*10的矩阵，利用sklearn的onehot编码
#y0 = 2 --> [0,1,0...0],y0 = 0 -->[0,...,0,1] ，0代表10

encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
print('y_onehot=',y_onehot.shape)
print('y[0]=',y[0],',y_onehot[0,:]=',y_onehot[0,:])

input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

print(cost(theta1,theta2,input_size,hidden_size,num_labels,X,y_onehot,learning_rate))

#带正则化的损失函数
def costReg(theta1,theta2,input_size,hidden_size,num_labels,X,y,learning_rate):
    m = X.shape[0]  #m=5000
    X = np.matrix(X)
    y = np.matrix(y)

    #计算前向传播
    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)

    #计算代价函数,y与h（假设函数）的预测值一一对应
    J = 0
    for i in range(m):
        first_term = np.multiply((-y[i,:]),np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J += np.sum(first_term-second_term)
    
    J = J/m
    #记住，正则化惩罚从小标1开始，所以theta0 不考虑！
    J += (float(learning_rate)/(2*m))*(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))
    return J

print(costReg(theta1,theta2,input_size,hidden_size,num_labels,X,y_onehot,learning_rate))

#sigmoid梯度
#g'(z) = d g(z)/dz = g(z)(1-g(z))
#g(z)=sigmoid(z)=1/(1+e^-z)
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))

print(sigmoid_gradient(0))

#随机初始化，训练神经网络时，将theta设定为(-e,e)之间的随机值，此处设定e=0.12
#np.random.random(size) 返回size大小的0-1随机浮点数
params=(np.random.random(size=hidden_size*(input_size+1)+num_labels*(hidden_size+1))-0.5)*0.24

#反向传播
#步骤为：给定训练集（x，y），先计算正向传播h(x),再对于l层的每个节点j，计算误差项detaj，这个数据衡量这个节点对最后输出的误差“贡献”多少
#输出节点，直接计算输出值和目标值的差值
#隐藏节点，基于现有权重和（l+1）层的误差，计算
def backprop(params,input_size,hidden_size,num_labels,X,y,learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)

    #参数重新构造矩阵???
    theta1 = np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
    theta2 = np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))

    #初始化
    J = 0
    delta1 = np.zeros(theta1.shape)     #(25,401)
    delta2 = np.zeros(theta2.shape)     #(10,26)

    #计算代价函数
    for i in range(m):
        first_term = np.multiply(-y[i,:],np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J += np.sum(first_term-second_term)

    J = J/m

    #反向传播计算，还不够清晰，得再理一遍
    for t in m:
        a1t = a1[t,:] #(1,401)
        z2t = z2[t,:] #(1,25)
        a2t = a2[t,:] #(1,26)
        ht = h[t,:]   #(1,10)
        yt = y[t,:]   #(1,10)

        d3t = ht - yt   #(1,10)

        z2t = np.insert(z2t,0,values=np.ones(1))   #(1,26)
        d2t = np.multiply((theta2.T * d3t.T).T,sigmoid_gradient(z2t))   #(1,26)
        
        #上一层的delta 等于这一层的a乘以下一层的误差
        delta1 = delta1 + (d2t[:,1:]).T*a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    return J , delta1,delta2

#梯度校验：把两个参数连接成一个长向量，f(theta) 约等于 (J(theta-e)+J(theta+e)) / 2e

#加入正则化的神经网络
def backpropReg(params,input_size,hidden_size,num_labels,X,y,learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    #参数重新构造矩阵???
    theta1 = np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
    theta2 = np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))

    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)

    #初始化
    J = 0
    delta1 = np.zeros(theta1.shape)     #(25,401)
    delta2 = np.zeros(theta2.shape)     #(10,26)

    #计算代价函数
    for i in range(m):
        first_term = np.multiply(-y[i,:],np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J += np.sum(first_term-second_term)

    J = J/m
    #加入正则项
    J += (float(learning_rate)/(2*m))*(np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))

    #反向传播计算，还不够清晰，得再理一遍
    for t in range(m):
        a1t = a1[t,:] #(1,401)
        z2t = z2[t,:] #(1,25)
        a2t = a2[t,:] #(1,26)
        ht = h[t,:]   #(1,10)
        yt = y[t,:]   #(1,10)

        d3t = ht - yt   #(1,10)

        z2t = np.insert(z2t,0,values=np.ones(1))   #(1,26)
        d2t = np.multiply((theta2.T * d3t.T).T,sigmoid_gradient(z2t))   #(1,26)
        
        #上一层的delta 等于这一层的a乘以下一层的误差
        delta1 = delta1 + (d2t[:,1:]).T*a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    #加入正则项,不惩罚theta0
    delta1[:,1:] = delta1[:,1:]+(theta1[:,1:]*learning_rate)
    delta2[:,1:] = delta2[:,1:]+(theta2[:,1:]*learning_rate)

    #concatenate 进行数组拼接
    grad = np.concatenate((np.ravel(delta1),np.ravel(delta2)))

    return J , grad

#使用工具库计算参数最优解
from scipy.optimize import minimize

fmin = minimize(fun=backpropReg,x0=(params),args=(input_size,hidden_size,num_labels,X,y_onehot,learning_rate),
                method='TNC',jac=True,options={'maxiter':250})
print(fmin)

X = np.matrix(X)
thetafinal1 = np.matrix(np.reshape(fmin.x[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
thetafinal2 = np.matrix(np.reshape(fmin.x[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))

#计算使用优化后的theta进行预测
a1,z2,a2,z3,h = forward_propagate(X,thetafinal1,thetafinal2)
y_pred = np.array(np.argmax(h,axis=1)+1)
print(y_pred)

#预测值和实际值比较
from sklearn.metrics import classification_report
print(classification_report(y,y_pred))

#可视化隐藏层
hidden_layer = thetafinal1[:,1:]
print(hidden_layer.shape)   #(25,400)

fig,ax_array = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True,figsize=(12,12))
for r in range(5):
    for c in range(5):
        ax_array[r,c].matshow(np.array(hidden_layer[5*r+c].reshape((20,20))),cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
