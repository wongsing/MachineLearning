#线性可分SVM
#观察C取值对决策边界的影响
import numpy as np 
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('D:\Pytest\MachineLearning\ML_test\SVM支持向量机\ex6data1.mat')

# print(data)
print(data.keys())

X,y=data['X'],data['y']
print(X.shape,y.shape)  #(51,2),(51,1) ,两个特征

#c=y.flatten()，表示用y来区分颜色，0/1
def plot_data():
    plt.scatter(X[:,0],X[:,1],c=y.flatten(),cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

plot_data()

#训练SVM分类器
from sklearn.svm import SVC
svc1 = SVC(C=1,kernel='linear')
#数据拟合
svc1.fit(X,y.flatten()) 
print(svc1.fit(X,y.flatten()))
#预测
svc1.predict(X)
print(svc1.predict(X))
#得分
svc1.score(X,y.flatten())
print(svc1.score(X,y.flatten()))
#画出决策曲线
#np.meshgrid 生成网格点坐标矩阵
#np.linspace(x_min,x_max,500) 生成连续的数据
#np.c_中的c 是 column(列)的缩写，就是按列叠加两个矩阵，就是把两个矩阵左右组合，要求行数相等
#plt.contour 画等高线的函数
def plot_boundary(model):
    x_min,x_max = -0.5,4.5
    y_min,y_max = 1.3,5
    xx,yy = np.meshgrid(np.linspace(x_min,x_max,500),np.linspace(y_min,y_max,500))
    z = model.predict(np.c_[xx.flatten(),yy.flatten()])
    print(xx.shape)
    zz = z.reshape(xx.shape)
    plt.contour(xx,yy,zz)

plot_boundary(svc1)
plot_data()
#结果显示有一个 样本值是错误分类，由于C较小，所以SVM有一定的容错率

#将C增大，会将离散点正确分类，但不是很好的决策曲线
svc100 = SVC(C=100,kernel='linear')
svc100.fit(X,y.flatten())
svc100.predict(X)
svc100.score(X,y.flatten())
print(svc100.score(X,y.flatten()))
plot_boundary(svc100)
plot_data()


