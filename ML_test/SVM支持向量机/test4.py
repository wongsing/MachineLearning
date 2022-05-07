#判断一封邮件是否为垃圾邮件
#X是包含多个关键词的特征向量,用0/1表示是否出现过
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC


#训练集数据
data1 = sio.loadmat('D:\Pytest\MachineLearning\ML_test\SVM支持向量机\spamTrain.mat')
print('data1:',data1.keys())
X,y = data1['X'],data1['y']
print(X.shape,y.shape)  #(4000,1899),(4000,1) 
print(data1)
#测试集数据
data2 = sio.loadmat('D:\Pytest\MachineLearning\ML_test\SVM支持向量机\spamTest.mat')
print('data2:',data2.keys())
Xtest,ytest = data2['Xtest'],data2['ytest']

#采用线性核
#选取最佳的C
Cvalues = [3,10,30,100,0.01,0.03,0.1,0.3,1]

best_score = 0
best_param = 0

for c in Cvalues:
    svc = SVC(C=c,kernel='linear')
    svc.fit(X,y.flatten())
    score = svc.score(Xtest,ytest.flatten())
    if(score > best_score):
        best_score = score
        best_param = c
print(best_score,best_param)    #(0.99,0.03)

svc1 = SVC(C=0.03,kernel='linear')
svc1.fit(X,y.flatten())
score_train = svc1.score(X,y.flatten())
score_test = svc1.score(Xtest,ytest.flatten())

print(score_train,score_test)