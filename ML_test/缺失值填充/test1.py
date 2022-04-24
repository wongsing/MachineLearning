#pandas中处理缺失值-->pd.NaT/np.nan
#删除 用 dropna== DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

from sqlite3 import Timestamp
import numpy as np
import pandas as pd

df = pd.DataFrame({"name":['wxy','www','zzz'],
                    "toy":[np.nan,'Bat','Tencent'],
                    "born":[pd.NaT,pd.Timestamp("1999-05-25"),pd.NaT]})

print(df)
print("*******删除缺失值后***********")
print(df.dropna())
print("*******删除缺失值后*指定行0、列1**********")
print(df.dropna(axis=1))
print(df.dropna(axis=0))
print("*******删除缺失值后*只有所有值才删**********")
print(df.dropna(how='all'))
print("*******删除缺失值后*只有出现两个空值才删**********")
print(df.dropna(thresh=2))
print("*******删除缺失值后*指定某个分组（列或者标签）含缺失值的行列**********")
print(df.dropna(subset=['name','born']))

