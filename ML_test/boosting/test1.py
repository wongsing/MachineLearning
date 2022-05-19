#利用决策树作为弱分类器，来实现Adaboost
#Adaboost:不断训练弱分类器，来针对性的提升前一轮中预测错误样本的权重
#最后通过加权所有弱分类器的训练结果得到最终的分类标签
#在sklearn库内置的Adaboost算法中，当解决分类问题时弱学习器选择最大深度为1的决策树（俗称决策树桩）
# 解决回归问题时则选择最大深度为3的决策树（CART）

import numpy as np
from sklearn import tree

#决策树桩
class DecisionTreeClassifierWithWeight:
    def __init__(self):
        self.best_err = 1 #最小的加权错误率
        self.best_fea_id = 0 #最优的特征id
        self.best_thres = 0 #选定特征的最优阀值
        self.best_op = 1 #阀值符号，其中1：>,0：<   --->代表的就像是正反向

    def fit(self,X,y,sample_weight = None):
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)    #初始权重1/N
        n = X.shape[1]  #列数
        for i in range(n):
            feature = X[:,i]
            fea_unique = np.sort(np.unique(feature))    #将所有特征值从小到大排序
            for j in range(len(fea_unique)-1):
                thres = (fea_unique[j]+fea_unique[j+1])/2   #逐一设置可能阀值
                for op in (0,1):
                    y_ = 2*(feature >= thres)-1 if op==1 else 2*(feature<thres)-1 #判断何种符号为最优
                    err = np.sum((y_ != y )*sample_weight)
                    if err < self.best_err: #当前的参数组合可以获得更低的错误率，更新最优参数
                        self.best_err = err
                        self.best_op = op
                        self.best_fea_id = i
                        self.best_thres = thres
        return self
    
    def predict(self,X):
        featrue = X[:,self.best_fea_id]
        return 2*(featrue >= self.best_thres)-1 if self.best_op == 1 else 2*(featrue < self.best_thres)-1
    
    def score(self,X,y,sample_weight=None):
        y_pre = self.predict(X)
        if sample_weight is not None:
            return np.sum((y_pre==y)*sample_weight)
        return np.mean(y_pre==y)

#使用的是sklearn库中的乳腺癌二分类数据
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X,y = load_breast_cancer(return_X_y=True)
print(X,y)
y = 2*y-1   #将0/1值映射为-1/1
print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y)

#对决策树桩进行训练和评分
print(DecisionTreeClassifierWithWeight().fit(X_train,y_train).score(X_test,y_test))



#Adaboost集成分类器
#n_estimators:选择弱分类器的个数
class AdaBoostClassifier_:
    def __init__(self,n_estimators = 50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
    
    def fit(self,X,y):
        sample_weight = np.ones(len(X))/len(X)  #初始权重为1/N
        for _ in range(self.n_estimators):
            dtc = DecisionTreeClassifierWithWeight().fit(X,y,sample_weight) #训练弱分类器
            alpha = 1/2 * np.log((1-dtc.best_err)/dtc.best_err) #权重系数
            y_pred = dtc.predict(X)
            sample_weight  *= np.exp(-alpha*y_pred*y)   #更新迭代样本权重
            sample_weight /= np.sum(sample_weight*np.exp(-alpha*y_pred*y)) #样本权重归一化，而李航书中提到的是 规范化因子不过不同
            self.estimators.append(dtc)
            self.alphas.append(alpha)
        return self

    def predict(self,X):
        y_pred = np.empty((len(X),self.n_estimators))   #预测结果二维数组，其中每一列代表一个弱学习器的预测结果
        for i in range(self.n_estimators):
            y_pred[:,i] = self.estimators[i].predict(X)
        y_pred = y_pred * np.array(self.alphas) #将预测结果与训练权重乘积作为预测结果
        return 2*(np.sum(y_pred, axis=1)>0)-1  # 以0为阈值，判断并映射为-1和1

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred==y)
    
from sklearn.ensemble import AdaBoostClassifier
score1= AdaBoostClassifier_().fit(X_train, y_train).score(X_test, y_test) # 0.986013986013986 
score2= AdaBoostClassifier().fit(X_train, y_train).score(X_test, y_test) # 0.965034965034965
print(score1,score2)


