# 随机森林回归进行填补

# 特征T不缺失的值对应的其他n-1个特征 + 本来的标签：X_train
# 特征T不缺失的值：Y_train
# 特征T缺失的值对应的其他n-1个特征 + 本来的标签：X_test
# 特征T缺失的值：未知，我们需要预测的Y_test

# 对于一个有n个特征的数据来说，其中特征T有缺失值，我们就把特征T当作标签，其他的n-1个特征和原本的标签组成新的特征矩阵。
# 那对于T来说，它没有缺失的部分，就是我们的Y_test，这部分数据既有标签也有特征.
# 而它缺失的部分，只有特征没有标签，就是我们需要预测的部分
#对于某一个特征大量缺失，其他特征却很完整的情况，非常适用

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.DataFrame({'Country':[12,34,23,45,34,23,12,2,3], 
 
                   'Income':[10000, 10000, 5000, 5002, 40000, 50000, 8000, 5000,3000],
 
                    'Age':[50, 43, 34, 40, 25, 25, 45, 32,12],'填充列':[2,4,7,4,5,np.nan,np.nan,np.nan,np.nan]})

print(df)

#数据分成两部分，包含缺失值的所有列，不含缺失值的列
df_full = df.drop(labels='填充列',axis=1)
print(df_full)

df_nan = df.loc[:,'填充列']
print(df_nan)

#区别测试集合训练集
Ytrain = df_nan[df_nan.notnull()]
Ytest = df_nan[df_nan.isnull()]
Xtrain = df_full.iloc[Ytrain.index]
Xtest = df_full.iloc[Ytest.index]

#实例化，用随机森林回归填补缺失值,第一步确定森林中树的数目，fit训练特征和标签（预测值），然后再通过predict预测特征测试集
rfc = RandomForestRegressor(n_estimators=100)
rfc = rfc.fit(Xtrain,Ytrain)
Ypredict = rfc.predict(Xtest)

print(Ypredict)

df_nan[df_nan.isnull()] = Ypredict
print(df)
