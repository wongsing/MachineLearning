# 习题8.1
#某公司招聘职员考查身体、业务能力、发展潜力这3项。已知10个人的数据。
#身体分为合格1、不合格0两级，
#业务能力、发展潜力分为上1、中2、下3三级。
#分类为合格1 、不合格-1两类
#假设弱分类器为决策树桩。试用AdaBoost算法学习一个强分类器。  


#加载数据
import numpy as np 
X = np.array([[0,1,3],[0,3,1],[1,2,2],[1,1,3],[1,2,3],[0,1,2],
            [1,1,2],[1,1,1],[1,3,1],[0,2,1]])
y = np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1])

print(X.shape)
m,n = X.shape
print(m,n)
#直接使用Adaboostclassifier分类器实现,默认采用CART决策树弱分类器
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X,y)
y_pred = clf.predict(X)
score = clf.score(X,y)
print('原始输出：',y)
print('预测输出：',y_pred)
print('预测正确率：{:.2%}'.format(score))

#自编程
#自定义Adaboost步骤
#1.给每个训练样本分配权重，初试权重w1 = 1/N
#2.针对带有权值的样本进行训练，得到模型Gm，初始模型为G1
#3.计算模型Gm的误分率em --记得要根据方向
#4.计算模型Gm的系数 alpha = 0.5*log[(1-em)/em]
#5.利用误分率em和规范化因子 更新权重
#6.计算组合模型fx的误分率 f(x)= 累加 am*Gm
#7.组合模型的误分率或迭代次数低于一定阀值，停止迭代 否则放回步骤二

class AdaBoost:
    def __init__(self,X,y, tol = 0.5 , max_iter = 10):
        #训练数据 实例
        self.X = X 
        #训练数据 标签
        self.y = y
        #训练终止条件 right_rate > self.tol
        self.tol = tol
        #最大迭代次数
        self.max_iter = max_iter
        #初始化样本权重w
        self.w = np.full((X.shape[0]),X.shape[0])
        #弱分类器
        self.G = []

    def bulid_stump(self):
        """
        以带权重的分类误差最小为目标，选择最佳分类阈值
        best_stump['dim'] 合适的特征所在维度
        best_stump['thresh']  合适特征的阈值
        best_stump['ineq']  树桩分类的标识lt,rt
        """
        m,n = np.shape(self.X)  #10 , 3
        #分类误差
        e_min = np.inf  #表示+∞,是没有确切的数值的
        #小于分类阀值的样本属于的标签类别
        sign = None
        #最优分类树桩
        best_stump={}
        for i in range(n):
            range_min = self.X[:,i].min()   #求每一种特征的最大最小值
            range_max = self.X[:,i].max()
            step_size = (range_max-range_min)/n
            for j in range(-1,int(n)+1):
                thresh_val = range_min + j*step_size  
                #计算左子树和右子树的误差
                for inequal in ['lt','rt']:
                    predict_vals = self.base_estimator(self.X,i,thresh_val,inequal)
                    err_arr = np.array(np.ones(m))
                    err_arr[predict_vals.T == self.y.T] = 0 
                    weighted_error = np.dot(self.w,err_arr)
                    if weighted_error < e_min:
                        e_min = weighted_error
                        sign = predict_vals
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh_val
                        best_stump['ineq'] = inequal
        print(sign)     
        return best_stump,sign,e_min

    def update_w(self,alpha,predict):
        #更新样本权重w
        P = self.w * np.exp(-alpha*self.y*predict)
        self.w = P/P.sum()  #P.sum()就是Z 规范因子

    @staticmethod
    def base_estimator(X,dimen,threshVal,threshIneq):
        #计算单个弱分类器（决策树桩）预测输出
        ret_array = np.ones(np.shape(X)[0]) #预测矩阵
        #左叶子，整个矩阵的样本进行比较赋值
        if threshIneq == 'lt':
            ret_array[X[:,dimen] <= threshVal] = -1.0
        else:
            ret_array[X[:,dimen] > threshVal] = -1.0
        return ret_array

    def fit(self):
        #对数据进行训练
        G = 0
        for i in range(self.max_iter):
            best_stump,sign,error = self.bulid_stump()  #获取当前迭代最佳分类阀值
            print(best_stump,sign,error)
            alpha = 1 / 2 * np.log((1-error)/error)     #获取本轮弱分类器的系数
            #弱分类器权重
            best_stump['alpha'] = alpha
            #保存弱分类器
            self.G.append(best_stump)
            #以下3行计算当前总分类器（之前所有的弱分类器加权和）分类效率
            G += alpha * sign   #sign就是预测值Gi
            y_predict = np.sign(G)  #sigh函数返回的是一个由 1 和 -1 组成的数组，表示原始值的符号
            error_rate = np.sum(np.abs(y_predict-self.y)) / 2 / self.y.shape[0]
            if error_rate < self.tol:   #满足终止条件，则跳出循环
                print('迭代次数：',i+1)
                break
            else:
                self.update_w(alpha,y_predict)  #若不满足，更新权重，继续迭代
    def predict(self , X):
        #对新数据进行预测
        m = np.shape(X)[0]
        G = np.zeros(m)
        for i in range(len(self.G)):
            stump = self.G[i]
            #遍历每一个弱分类器，进行加权
            _G = self.base_estimator(X,stump['dim'],stump['thresh'],stump['ineq'])
            alpha = stump['alpha']
            G += alpha*G
        y_predict = np.sign(G)
        return y_predict.astype(int)    #astype转类型

    def score(self,X,y):
        y_predict = self.predict(X)
        error_rate = np.sum(np.abs(y_predict-y))/2/y.shape[0]
        return 1-error_rate #返回正确率


clf = AdaBoost(X,y)
clf.fit()
y_predict = clf.predict(X,y)
score = clf.score(X,y)
print('原始输出：',y)
print('预测输出：',y_pred)
print('预测正确率：{:.2%}'.format(score))
    







