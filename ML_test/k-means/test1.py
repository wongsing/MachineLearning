from operator import index
from random import randrange
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data1 = sio.loadmat('D:\Pytest\MachineLearning\ML_test\k-means\ex7data2.mat')
print(data1.keys())

X = data1['X']
print(X.shape)  #(300,2)

plt.scatter(X[:,0],X[:,1])
plt.show()

#获取每个样本所属的类别
#linalg.norm :求范数!范数，在机器学习中通常用于衡量一个向量的大小(L2范数等于向量的模)||x||2
def find_centroids(X,centros):
    idx=[]

    for i in range(len(X)):
        #(2,),(k,2)-->广播机制=>(k,2)
        dist = np.linalg.norm((X[i]-centros),axis=1) #(k,) 
        # print('i=',i,'dist=',dist)
        id_i = np.argmin(dist)
        # print('id_i=',id_i)
        idx.append(id_i)
    
    return np.array(idx)

centros = np.array([[3,3],[6,2],[8,5]])
idx = find_centroids(X,centros)
print(idx[:3])  #0,2,1,idx就是类别数组
# print(idx.shape,idx)

#计算聚类中心点，该类的平均值，也就是聚类中心的移动
def compute_centros(X,idx,k):
    centros = []

    for i in range(k):
        centros_i = np.mean(X[idx == i],axis=0) #idx==i 就相当于找到索引,如data['X']
        centros.append(centros_i)
    
    return np.array(centros)

print(compute_centros(X,idx,k=3))

#运行kmeans，实现整个聚类过程,重复1、2过程
def run_kmeans(X,centros,iters):
    k = len(centros)
    centros_all = [] #记录所有聚类中心点
    centros_all.append(centros)
    centros_i = centros
    for i in range(iters):
        idx = find_centroids(X,centros_i)
        centros_i = compute_centros(X,idx,k)
        centros_all.append(centros_i)
    
    return idx,np.array(centros_all)
#由于每次迭代都获得三个类别的聚类中心点坐标，迭代10次就相当于(10,3,2)


#绘制数据集合聚类中心的移动轨迹
def plot_data(X,centros_all,idx):
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=idx,cmap='rainbow')
    plt.plot(centros_all[:,:,0],centros_all[:,:,1],'kx--')
    plt.show()

idx,centros_all = run_kmeans(X,centros,iters=10)
plot_data(X,centros_all,idx)

#观察初始聚类点的位置对聚类效果的影响,由于初始位置选择不当，会出现局部最优值
#从样本中随机选取聚类点
def init_centros(X,k):
    index = np.random.choice(len(X),k)
    return X[index]

print(init_centros(X,k=3))

for i in range(4):
    idx,centros_all = run_kmeans(X,init_centros(X,k=3),iters=10)
    plot_data(X,centros_all,idx)
