#-*- coding: utf-8 -*-
#阿里天池预测项目
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as ACF
from statsmodels.tsa.stattools import pacf as PACF
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor

# 1数据提取
df = pd.read_table('../tmp/sample_tmp.txt',index_col='d')
df.index = pd.to_datetime(df.index)
df = df[['date','day_of_week','ifworkday','cnt']]
# print df.info(),df.index,df.head()

# 2数据预处理
train=df[(df.index>='2014-01-01')&(df.index<'2015-01-01')]#训练数据,取两年数据
original=df[(df.index>='2015-01-01')&(df.index<='2015-12-31')]#实际数据
# print train.shape,train.head()
# print original.shape,original.head()

# # 3数据建模
print train.describe()
count = train.groupby(['ifworkday','day_of_week'],as_index=False).mean()
print count
resulta=original[(original['ifworkday']==1)&(original['day_of_week']==6)].copy()
resulta.loc[:,'cnt_y']=230
resultb=original[original['ifworkday']==0].copy()
resultb.loc[:,'cnt_y']=2000
resultc=original[(original['ifworkday']==1)&(original['day_of_week']!=6)].copy()
resultc.loc[:,'cnt_y']=25

result=pd.concat([resulta,resultb],axis=0).sort_values('date')
result=pd.concat([result,resultc],axis=0).sort_values('date')
print result.shape,result.head()

# # 4数据模型评估
plt.figure(facecolor='white')
result['cnt'].plot(color='blue', label='Original')
result['cnt_y'].plot(color='red', label='Predict')
plt.legend(loc='best')
plt.title('MSE: %.4f' % mean_squared_error(result['cnt'],result['cnt_y']))
plt.show()

