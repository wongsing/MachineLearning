#案例2，使用kmeans对图片颜色进行聚类
#RGB图像，每个像素点值范围0-255 8bit存储 ，类别为16种 4bit

from matplotlib import image
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pylab 
data = sio.loadmat('D:\Pytest\MachineLearning\ML_test\k-means\\bird_small.mat')
print(data.keys())

A=data['A']
print(A.shape)

from skimage import io
image = io.imread('D:\Pytest\MachineLearning\ML_test\k-means\\bird_small.png')
plt.imshow(image)
pylab.show()

A = A /255 #对A进行标准化，全部化成0-1
A = A.reshape(-1,3)

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

#计算聚类中心点，该类的平均值，也就是聚类中心的移动
def compute_centros(X,idx,k):
    centros = []

    for i in range(k):
        centros_i = np.mean(X[idx == i],axis=0) #idx==i 就相当于找到索引,如data['X']
        centros.append(centros_i)
    
    return np.array(centros)

#运行kmeans，实现整个聚类过程,重复1、2过程
def run_kmeans(X,centros,iters):
    k = len(centros)
    centros_all = [] #记录所有聚类中心点
    centros_all.append(centros) #先加入初始点
    centros_i = centros
    for i in range(iters):
        idx = find_centroids(X,centros_i)
        centros_i = compute_centros(X,idx,k)
        centros_all.append(centros_i)
    
    return idx,np.array(centros_all)
#由于每次迭代都获得三个类别的聚类中心点坐标，迭代10次就相当于(10,3,2)

#从样本中随机选取聚类点
def init_centros(X,k):
    index = np.random.choice(len(X),k)
    return X[index]

k=16
idx , centros_all = run_kmeans(A,init_centros(A,k=16),iters=20)
centros = centros_all[-1]   #取的是最后一次迭代的聚类点中心位置
im = np.zeros(A.shape)
for i in range(k):
    im[idx == i] = centros[i]
im = im.reshape(128,128,3)
plt.imshow(im)
pylab.show()