
#填补缺失值 使用fillna()
#DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
import imp
import numpy as np
import pandas as pd

data  = pd.DataFrame([[np.nan,2,np.nan,0],
                        [3,4,np.nan,1],
                        [np.nan,np.nan,np.nan,5],
                        [np.nan,3,np.nan,4],
                        [np.nan,3,np.nan,4]],
                        columns=list('ABCD'))
print(data)

#ffill:缺失值前面的值对缺失值进行填充
#backfill/bfill:缺失值后面的一个值代替前面的缺失值
#用这种方法时不能与value同时出现,可以指定行列
print("*******填补缺失值后*ffill指定行0、列1**********")
print(data.fillna(axis=1,method='ffill'))
print(data.fillna(axis=0,method='ffill'))

print("*******填补缺失值后*指定常量**********")
print(data.fillna(0))

print("*******填补缺失值后* 不同列填不同值，行还需研究**********")
trans1={"A":9,"B":8,"C":7,"D":6}
print(data.fillna(value=trans1))

print("*******填补缺失值后* 不同列填不同值，但有限制，限制替换一次**********")
trans1={"A":9,"B":8,"C":7,"D":6}
print(data.fillna(value=trans1,limit=1))

print("*******填补缺失值后*采用mean均值**********")
print(data.fillna(data.mean()['A':'B']))
print(data.fillna(data.mean()))



