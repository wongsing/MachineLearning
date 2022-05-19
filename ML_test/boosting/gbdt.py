from random import random
from string import digits
from unicodedata import digit
from numpy import nper, percentile
from sklearn import tree
#gbdt：梯度提升决策树，采用的是CART树，可以用来回归和分类
#选择特征、构建特征
#利用负梯度进行下降，使得损失函数极小化

#GBDT的工具包：GradientBoostingClassifier，XGBoost、lightGBM

#加载数据
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

#划分数据
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#构建模型
'''
调参：
loss：损失函数。有deviance和exponential两种。deviance是采用对数似然，exponential是指数损失，后者相当于AdaBoost。
n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管
由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
max_depth：CART最大深度，默认为None
min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
min_leaf_nodes：最大叶子节点数
'''

from sklearn.ensemble import GradientBoostingClassifier
dtc = GradientBoostingClassifier(loss='deviance',learning_rate=0.005,n_estimators=100,
                                subsample=1,min_samples_split=2,
                                min_samples_leaf=1,min_weight_fraction_leaf=0,
                                max_depth=3,init=None,random_state=None,
                                max_features=None,verbose=0,
                                max_leaf_nodes=None,warm_start=False,
                                presort='auto')

#训练模型
dtc.fit(X_train,y_train)
#测试
y_pred = dtc.predict(X_test)
#检验模型
print('Model in train score is:',dtc.score(X_train,y_train))
print('Model in test score is:',dtc.score(X_test,y_test))

from sklearn.metrics import classification_report
print('report is :',classification_report(y_test,y_pred))


#简易版的GBDT：梯度提升决策树
import numpy as np
np.random.seed(42)
X = np.random.rand(100,1)-0.5
y = 3*X[:,0]**2 + 0.05*np.random.randn(100)
#弱学习器采用决策树
from sklearn.tree import DecisionTreeRegressor
#第一个模型
tree_reg1 = DecisionTreeRegressor()
tree_reg1.fit(X,y)
#残差
y2 = y-tree_reg1.predict(X)
#训练第二个模型
tree_reg2 = DecisionTreeRegressor()
tree_reg2.fit(X,y2)
#残差
y3 = y2-tree_reg2.predict(X)
#训练第三个模型
tree_reg3 = DecisionTreeRegressor()
tree_reg3.fit(X,y3)
#测试
X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in(tree_reg1 , tree_reg2,tree_reg3))

print(y_pred)








