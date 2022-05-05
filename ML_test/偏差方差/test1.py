#了解算法性能中的偏差和方差的概念
#偏差：预测值和真实值之间的差距，也就是算法本身的拟合程度
#方差：预测值的变化范围，数据扰动所造成的影响

#训练集：训练模型
#验证集：模型选择，模型的最大优化
#测试集：利用训练好的模型来测试其泛化能力

#案例：利用水库水位变化预测大坝出水量

from cProfile import label
from cgi import test
from re import X
import matplotlib.pyplot as plt
import numpy as np 
from scipy.io import loadmat
from scipy.optimize import minimize

#进行数据可视化
data = loadmat('D:\Pytest\MachineLearning\ML_test\偏差方差\ex5data1.mat')
print(data.keys())

#训练集
X_train,y_train = data['X'],data['y']
print(X_train.shape,y_train.shape)  #(12,1) 12个样本，1个特征

#验证集
X_val,y_val = data['Xval'],data['yval']
print(X_val.shape,y_val.shape)

#测试集
X_test,y_test = data['Xtest'],data['ytest']
print(X_test.shape,y_test.shape)

#添加x0=1
X_train = np.insert(X_train,0,1,axis=1)
X_val = np.insert(X_val,0,1,axis=1)
X_test = np.insert(X_test,0,1,axis=1)

def plot_data():
    fig,ax = plt.subplots()
    ax.scatter(X_train[:,1],y_train)
    ax.set(xlabel='change in water level(x)',
            ylabel='water flowing out og the dam(y)')
plot_data()

#线性回归 h(x) = @0+@1*x
#带正则化的损失函数--用@进行矩阵运算， 不需要将theta进行转置后相乘
def reg_cost(theta,X,y,lamda):
    cost = np.sum(np.power((X@theta-y.flatten()),2))
    reg = theta[1:]@theta[1:]*lamda

    return(cost+reg)/(2*len(X))

theta = np.ones(X_train.shape[1])   #theta初始值为[1,1]
print(theta)
lamda = 1
print(reg_cost(theta,X_train,y_train,lamda))

#梯度
def reg_gradient(theta,X,y,lamda):
    grad = (X@theta-y.flatten())@X  #由于梯度是一个维向量，不需要用sum
    reg = lamda * theta
    reg[0] = 0  #正则化不惩罚0项

    return (grad+reg)/len(X)

print(reg_gradient(theta,X_train,y_train,lamda))

#计算最佳参数
def train_model(X,y,lamda):
    theta = np.ones(X.shape[1])
    res = minimize(fun= reg_cost,
                    x0= theta,
                    args=(X,y,lamda),
                    method='TNC',
                    jac=reg_gradient)
    return res.x

theta_final = train_model(X_train,y_train,lamda=0)
plot_data()
plt.plot(X_train[:,1:],X_train@theta_final,c='r')
plt.show()  #结果呈现欠拟合状态，偏差非常大


#训练样本从1开始递增训练，比较训练集和验证集上的损失函数的变换情况
def plot_learning_curve(X_train,y_train,X_val,y_val,lamda):
    x = range(1,len(X_train)+1)
    training_cost = []
    cv_cost = []

    for i in x:
        res = train_model(X_train[:i,:],y_train[:i,:],lamda)
        training_cost_i = reg_cost(res,X_train[:i,:],y_train[:i,:],lamda)
        cv_cost_i = reg_cost(res,X_val,y_val,lamda)
        training_cost.append(training_cost_i)
        cv_cost.append(cv_cost_i)

    plt.plot(x,training_cost,label='training cost')
    plt.plot(x,cv_cost,label='cv cost')
    plt.legend()
    plt.xlabel('number of training examples')
    plt.ylabel('error')
    plt.show()

plot_learning_curve(X_train,y_train,X_val,y_val,lamda=0)
#结果呈现欠拟合状态，高偏差，由简单的线性模型所造成的

#计算Jtrain(@)和Jcv(@)，如果同时都很大，就遇到高偏差问题，如果Jcv(@)>>Jtrain(@)遇到高方差问题
#高方差：采取更多的样本数据 / 减少特征数量（去非主特征） /  增加正则化参数
#高偏差：引入更多的相关特征 / 采用多项式特征 / 减少正则化参数

#本案例中，只有一个特征：水库水位，并且之前的正则化参数lamda取0，所以只能考虑采用多项式特征
#多项式特征：提高幂次-->h(x) = @0+@1*x+@2*x^2+..+ (用循环)
def poly_feature(X,power):
    for i in range(2,power+1):
        X = np.insert(X,X.shape[1],np.power(X[:,1],i),axis=1)
    return X

#进行归一化，用训练集的均值和方差,按行
def get_means_stds(X):
    means = np.mean(X,axis=0)
    stds = np.std(X,axis=0)
    return means,stds
#第一列不进行归一化
def feature_normalize(X,means,stds):
    X[:,1:] = (X[:,1:] - means[1:]) / stds[1:]
    return X

power = 6

#做多项式特征
X_train_poly = poly_feature(X_train,power)
X_val_poly = poly_feature(X_val,power)
X_test_poly = poly_feature(X_test,power)
#求方差、均值
train_means,train_stds = get_means_stds(X_train_poly)
X_train_norm = feature_normalize(X_train_poly,train_means,train_stds)
X_val_norm = feature_normalize(X_val_poly,train_means,train_stds)
X_test_norm = feature_normalize(X_test_poly,train_means,train_stds)

theta_fit  = train_model(X_train_norm,y_train,lamda=0)

def plot_poly_fit():
    plot_data()

    x = np.linspace(-60,60,100)
    xx = x.reshape(100,1)
    xx = np.insert(xx,0,1,axis=1)
    xx = poly_feature(xx,power)
    xx = feature_normalize(xx,train_means,train_stds)

    plt.plot(x,xx@theta_fit,'r--')
    plt.show()

plot_poly_fit() #训练集表现

#观察正则化参数，可以改变模型拟合状态，lamda过小的话模型可能过拟合，lamda过大的话模型可能前拟合
plot_learning_curve(X_train_norm,y_train,X_val_norm,y_val,lamda=0)   #训练集和验证集的表现

plot_learning_curve(X_train_norm,y_train,X_val_norm,y_val,lamda=1) 

plot_learning_curve(X_train_norm,y_train,X_val_norm,y_val,lamda=100) 

#正则化参数的选取，选最优
lamdas = [0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
training_cost = []
cv_cost = []

for lamda in lamdas:
    res = train_model(X_train_norm,y_train,lamda)
    #lamda我们只在训练时用的
    tc = reg_cost(res,X_train_norm,y_train,lamda=0)
    cv = reg_cost(res,X_val_norm,y_val,lamda=0)

    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(lamdas,training_cost,label='training cost')
plt.plot(lamdas,cv_cost,label='cv cost')
plt.legend()
plt.show()

print(lamdas[np.argmin(cv_cost)])

#看看最小值在测试集的效果/误差
res = train_model(X_train_norm,y_train,lamda=3)
test_cost = reg_cost(res,X_test_norm,y_test,lamda=0)
print(test_cost)