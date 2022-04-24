#sklearn多变量填充

# 在每个步骤中，将特征列指定为输出y，将其他特征列视为输入X。
# 把回归器拟合到已知的y的（X，y）上。
# 然后，使用回归器预测y的缺失值。
# 针对每个特征以迭代方式完成此操作，然后在max_iter插补回合中重复此操作。
# 返回最后一轮估算的结果。
 

from random import random
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


imp = IterativeImputer(max_iter = 10,random_state = 0)
imp.fit([[1,2],[3,6],[4,8],[np.nan,3],[7,np.nan]])  #类似回归器进行拟合
# print(IterativeImputer(random_state = 0))

X_test = [[np.nan,2],[6,np.nan],[np.nan,6]]

#保存精度
print(np.round(imp.transform(X_test)))