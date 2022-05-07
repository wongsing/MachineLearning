#线性不可分:SVM利用核函数，使得将低维空间映射到高维空间,可以在低维空间计算出高维空间的点积结果
#核函数：h(x)=@0 + @1*f1 + @2*f2 + @3*f3+....
#核函数：多项式核、线性核、高斯核、拉普拉斯核、sigmoid核

#本练习使用高斯核函数，Gamma越小，模型复杂度越低；Gamma越大，模型复杂度就越高

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = sio.loadmat('D:\Pytest\MachineLearning\ML_test\SVM支持向量机\ex6data2.mat')

print(data.keys())

X,y = data['X'],data['y']
print(X.shape,y.shape)  #(863,2),(863,1) 2个特征

def plot_data():
    plt.scatter(X[:,0],X[:,1],c=y.flatten(),cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()
plot_data()

#训练SVM
svc1 = SVC(C=1,kernel='rbf',gamma=100)
svc1.fit(X,y.flatten())
svc1.score(X,y.flatten())
print(svc1.score(X,y.flatten()))

#绘制决策曲线
def plot_boundary(model):
    x_min,x_max = 0,1
    y_min,y_max = 0.4,1
    xx,yy = np.meshgrid(np.linspace(x_min,x_max,500),np.linspace(y_min,y_max,500))
    z = model.predict(np.c_[xx.flatten(),yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx,yy,zz)

plot_boundary(svc1)
plot_data()
#当gamma=1，拟合度很低，准确率低，模型复杂度低
#当gamma = 50、100 ，准确率高，模型复杂度也高
