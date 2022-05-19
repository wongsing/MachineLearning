import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt

#Python类中规定，函数的第一个参数是实例对象本身，并且约定俗成，把其名字写为self。
# 其作用相当于java中的this，表示当前类的对象，可以调用当前类中的属性和方法。

# data
def create_data():
    iris = load_iris()  #鸢尾花数据集:'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    # print(iris)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target   # ['setosa' 'versicolor' 'virginica']
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    #按行索引，取出第0列第1列和最后一列，即取出sepal长度、宽度和标签
    #X为sepal length，sepal width y为标签 
    data = np.array(df.iloc[:100, [0, 1, -1]])  #只选取第0/1列和最后一列,前100个
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    #X为sepal length，sepal width , y为标签 
    return data[:,:2], data[:,-1]

X, y = create_data()
# train_test_split函数用于将矩阵随机划分为训练子集和测试子集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.scatter(X[:50,0],X[:50,1], label='0')
plt.scatter(X[50:,0],X[50:,1], label='1')
plt.legend()
plt.show()

#learningrate : 无论是回归还是分类，都是每个弱学习期的权重缩减系数
#使用sklearn
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100,learning_rate=0.5)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

#自定义Adaboost步骤
#1.给每个训练样本分配权重，初试权重w1 = 1/N
#2.针对带有权值的样本进行训练，得到模型Gm，初始模型为G1
#3.计算模型Gm的误分率em --记得要根据方向
#4.计算模型Gm的系数 alpha = 0.5*log[(1-em)/em]
#5.利用误分率em和规范化因子 更新权重
#6.计算组合模型fx的误分率 f(x)= 累加 am*Gm
#7.组合模型的误分率或迭代次数低于一定阀值，停止迭代 否则放回步骤二

class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def init_args(self, datasets, labels):

        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape
        # print(datasets.shape)
        # 弱分类器数目和集合
        self.clf_sets = []

        # 初始化weights
        self.weights = [1.0 / self.M] * self.M

        # G(x)系数 alpha
        self.alpha = []

    #弱分类器
    def _G(self, features, labels, weights):
        m = len(features)
        error = 100000.0  # 无穷大
        best_v = 0.0
        # 单维features
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min +
                  self.learning_rate) // self.learning_rate
        # // 表示整数除法,它可以返回商的整数部分(向下取整)
        # print('n_step:{}'.format(n_step))
        direct, compare_array = None, None
        #选取基本分类器，选取最佳的
        for i in range(1, int(n_step)):
            #选取阀值，
            v = features_min + self.learning_rate * i

            if v not in features:
                # 误分类计算，然后计算两类的误差率选取最小的,根据方向
                #正确的分成错误的，计算误差率
                compare_array_positive = np.array(
                    [1 if features[k] > v else -1 for k in range(m)])
                weight_error_positive = sum([
                    weights[k] for k in range(m)
                    if compare_array_positive[k] != labels[k]
                ])
                #错误的分成正确的
                compare_array_nagetive = np.array(
                    [-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum([
                    weights[k] for k in range(m)
                    if compare_array_nagetive[k] != labels[k]
                ])
                #然后计算两类的误差率选取最小的
                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'

                # print('v:{} error:{}'.format(v, weight_error))
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v  
        return best_v, direct, error, compare_array

    # 计算alpha
    def _alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    # 规范化因子
    def _Z(self, weights, a, clf):
        return sum([
            weights[i] * np.exp(-1 * a * self.Y[i] * clf[i])
            for i in range(self.M)
        ])

    # 权值更新
    def _w(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(
                -1 * a * self.Y[i] * clf[i]) / Z

    # G(x)的线性组合
    def _f(self, alpha, clf_sets):
        pass

    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

    def fit(self, X, y):
        self.init_args(X, y)
        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度, 选择误差最小的
            for j in range(self.N):
                features = self.X[:, j]
                # 分类阈值，分类误差，分类结果
                v, direct, error, compare_array = self._G(
                    features, self.Y, self.weights)

                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j

                # print('epoch:{}/{} feature:{} error:{} v:{}'.format(epoch, self.clf_num, j, error, best_v))
                if best_clf_error == 0:
                    break

            # 计算G(x)系数a
            a = self._alpha(best_clf_error)
            self.alpha.append(a)
            # 记录分类器
            self.clf_sets.append((axis, best_v, final_direct))
            # 规范化因子
            Z = self._Z(self.weights, a, clf_result)
            # 权值更新
            self._w(a, clf_result, Z)


#             print('classifier:{}/{} error:{:.3f} v:{} direct:{} a:{:.5f}'.format(epoch+1, self.clf_num, error, best_v, final_direct, a))
#             print('weight:{}'.format(self.weights))
#             print('\n')

    #累加
    def predict(self, feature):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)
        # sign
        return 1 if result > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1

        return right_count / len(X_test)

X = np.arange(10).reshape(10, 1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

clf = AdaBoost(n_estimators=3, learning_rate=0.5)
clf.fit(X, y)

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = AdaBoost(n_estimators=10, learning_rate=0.2)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# 100次结果
result = []
for i in range(1, 101):
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = AdaBoost(n_estimators=100, learning_rate=0.2)
    clf.fit(X_train, y_train)
    r = clf.score(X_test, y_test)
    # print('{}/100 score：{}'.format(i, r))
    result.append(r)

print('average score:{:.3f}%'.format(sum(result)))



