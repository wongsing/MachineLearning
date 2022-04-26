from cProfile import label
from math import degrees
from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
#正则化
#工厂主管，芯片的两次测试结果来判断，是否丢弃芯片
path = open('D:\Pytest\MachineLearning\ML_test\逻辑回归和正则化\ex2data2.txt')
data2 = pd.read_csv(path,header=None,names=['Test 1','Test 2','Accepted'])
print(data2.head())

#数据可视化
positive2 = data2[data2['Accepted'].isin([1])]
negative2 = data2[data2['Accepted'].isin([0])]

f,ax = plt.subplots(figsize = (10,8))
ax.scatter(positive2['Test 1'],positive2['Test 2'],s=50,c='b',marker='o',label='Accepted')
ax.scatter(negative2['Test 1'],negative2['Test 2'],s=50,c='r',marker='x',label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()

#特征映射，由于不好用直线进行分类，所以要提高特征的幂次
degree = 6
data = data2
x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3,'Ones',1)

#h(x)=g(@0+@1x1+@2x2+@3x1^2+@4x1x2+@5x2^2)....
for i in range (1,degree+1):
    for j in range(0,i+1):
        data['F'+str(i-j)+str(j)]=np.power(x1,i-j)*np.power(x2,j)

#上面的已经算出过这两列，去掉
data.drop('Test 1',axis=1,inplace=True)
data.drop('Test 2',axis=1,inplace=True)

print(data.head())

def sigmoid(z):
    return 1/(1+np.exp(-z))

#正则损失函数，拟合最重要看参数learningRate
def costReg(theta,X,y,learningRate ):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    reg = learningRate/(2*len(X))* np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second) / len(X) + reg

def gradientReg(theta,X,y,learningRate):
    theta = np.matrix(theta)
    X =  np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X*theta.T)-y

    for i in range(parameters):
        term = np.multiply(error,X[:,i])
        #因为，正则化对参数惩罚只从1开始
        if (i==0):
            grad[i] = np.sum(term)/len(X)
        else:
            grad[i] = (np.sum(term)/len(X)) + ((learningRate / len(X)) * theta[:,i])
        
    return grad

#初始化数据
cols = data.shape[1]
X2 = data.iloc[:,1:cols]
y2 = data.iloc[:,0:1]
theta2 = np.zeros(cols-1)

X2 = np.array(X2.values)
y2 = np.array(y2.values)

#初始化lamda λ =1
learningRate = 1
#初试代价
print(costReg(theta2,X2,y2,learningRate))

result2 = opt.fmin_tnc(func=costReg,x0=theta2,fprime=gradientReg,args=(X2,y2,learningRate))
print(result2)

#评价模型的准确率,返回list
def predict(theta,X):
    probability = sigmoid(X*theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result2[0])
predictions = predict(theta_min,X2)
correct = [1 if((a==1 and b==1)or(a==0 and b==0))else 0 for (a,b) in zip(predictions,y2)]
accuracy = (sum(map(int,correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

#画出决策曲线
def hfunc2(theta,x1,x2):    #假设函数，选了6次幂
    temp = theta[0][0]
    place = 0
    for i in range(1,degree+1):
        for j in range(0,i+1):
            temp += np.power(x1,i-j)*np.power(x2,j)*theta[0][place+1]
            place += 1
    return temp

#不太懂 记录一下，决策曲线
def find_decision_boundary(theta):
    t1 = np.linspace(-1,1.5,1000)   #-1到1.5 化成1000个数据点
    t2 = np.linspace(-1,1.5,1000)

    cordinates = [(x,y) for x in t1 for y in t2]
    x_cord,y_cord = zip(*cordinates)    #产生( )元组 tuple
    h_val = pd.DataFrame({'x1':x_cord,'x2':y_cord}) 
    # print(h_val)
    h_val['hval'] = hfunc2(theta,h_val['x1'],h_val['x2'])

    decision = h_val[np.abs(h_val['hval']) < 2 * 10**-3]
    return decision.x1 , decision.x2


f,ax = plt.subplots(figsize = (10,8))
ax.scatter(positive2['Test 1'],positive2['Test 2'],s=50,c='b',marker='o',label='Accepted')
ax.scatter(negative2['Test 1'],negative2['Test 2'],s=50,c='r',marker='x',label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x,y=find_decision_boundary(result2)
# print(x,y)
plt.scatter(x,y,c='y',s=10,label='Prediction')
ax.legend()
plt.show()

#考虑过拟合
learningRate2 = 0
result3 = opt.fmin_tnc(func = costReg , x0 = theta2,fprime= gradientReg,args=(X2,y2,learningRate2))
print(result3)

f,ax = plt.subplots(figsize=(10,8))
ax.scatter(positive2['Test 1'],positive2['Test 2'],s=50,c='b',marker='o',label='Acceoted')
ax.scatter(negative2['Test 1'],negative2['Test 2'],s=50,c='r',marker='x',label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x,y=find_decision_boundary(result3)
plt.scatter(x,y,c='y',s=10,label='Prediction')
ax.legend()
plt.show()

#考虑欠拟合
learningRate3 = 100
result4 = opt.fmin_tnc(func = costReg , x0 = theta2,fprime= gradientReg,args=(X2,y2,learningRate3))
print(result3)

f,ax = plt.subplots(figsize=(10,8))
ax.scatter(positive2['Test 1'],positive2['Test 2'],s=50,c='b',marker='o',label='Acceoted')
ax.scatter(negative2['Test 1'],negative2['Test 2'],s=50,c='r',marker='x',label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x,y=find_decision_boundary(result4)
plt.scatter(x,y,c='y',s=10,label='Prediction')
ax.legend()
plt.show()