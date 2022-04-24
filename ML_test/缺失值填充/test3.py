#sklearn库，关于缺失值填充的impute.SimpleImputer/随机森林回归
#sklearn.impute.SimpleImputer (missing_values=nan, strategy=’mean’, fill_value=None, verbose=0,copy=True)
#missing_values：缺失值长什么样
#strategy:采用什么方法填充，默认均值。mean/median/most_frequent（众数）/constant（常量），前两个数值型，后两个字符和数值均可
#copy:是否创建特征矩阵的副本，不然填补到原有特征矩阵中

#需要首先实例化方法

#单变量的插补

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
df  = pd.DataFrame([[np.nan,2,np.nan,0],
                        [3,4,np.nan,1],
                        [np.nan,np.nan,np.nan,5],
                        [np.nan,3,np.nan,4],
                        [np.nan,3,np.nan,4]],
                        columns=list('ABCD'))
print(df)
print("************均值处理*******")
df_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
df = df_mean.fit_transform(df)
print(df)
print(type(df))
#结果只保留可以求均值的列，不存在均值就会删除而且df是ndarray类型，不是dataFrame
d=pd.DataFrame(df)
print(type(d))

print("************常数处理*******")
df_0 = SimpleImputer(strategy='constant',fill_value=9)
df_cons = df_0.fit_transform(df)
d=pd.DataFrame(df_cons)
print(df_cons)
print(type(d))

print("************对不同列填补*******")
df = pd.DataFrame([[np.nan, 2, np.nan, 'a'],
                  [3, 4, np.nan, 'a'],
                 [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 4, np.nan, 'b']],
                 columns=list('ABCD'))
print(df)

df_mean = SimpleImputer(missing_values=np.nan,strategy='mean',copy=False)
df_median = SimpleImputer(missing_values=np.nan,strategy='median',copy=False)
df_0 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=9,copy=False)
df_most_frequent = SimpleImputer(missing_values=np.nan,strategy='most_frequent',copy=False)

#对不同列使用不同的方法
#reshape(-1,1)表示变成1列，(1,-1)变成1行，-1可以表示成未指定为给定的值，让计算机帮忙计算有几行/几列
df_A = df.loc[:,'A'].values.reshape(-1,1)
df.loc[:,'A'] = df_mean.fit_transform(df_A)

df_B = df.loc[:,'B'].values.reshape(-1,1)
df.loc[:,'B'] = df_median.fit_transform(df_B)

df_C = df.loc[:,'C'].values.reshape(-1,1)
df.loc[:,'C'] = df_0.fit_transform(df_C)

df_D = df.loc[:,'D'].values.reshape(-1,1)
df.loc[:,'D'] = df_most_frequent.fit_transform(df_D)
print(df)


