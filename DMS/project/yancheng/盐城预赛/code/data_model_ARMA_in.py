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
df = df[['ifworkday','cnt']]
# print df.info(),df.index,df.head()

# 2数据预处理
train=df[(df.index>='2013-01-01')&(df.index<'2015-01-01')]#训练数据,取两年数据
original=df[(df.index>='2015-01-01')&(df.index<='2015-12-31')]#实际数据
# print train.info(),train.index,train.head()
# print original.info(),original.index,original.head()

# 3数据建模
model = ARMA(train['cnt'], order=(6,1)).fit( disp=-1)# ARMA_model
trainA=train[train['ifworkday']==0]
trainB=train[train['ifworkday']==1]
modelA = ARMA(trainA['cnt'], order=(21,10)).fit( disp=-1)# ARMA_model
modelB = ARMA(trainB['cnt'], order=(1,1)).fit( disp=-1)# ARMA_model

# 4数据模型评估
# 4.1样本内评估
resultin=pd.DataFrame(model.predict(start=0,end=train.shape[0]-1),columns=['resultin'])
resultinA=pd.DataFrame(modelA.predict(start=0,end=trainA.shape[0]-1),columns=['resultin'])
resultinB=pd.DataFrame(modelB.predict(start=0,end=trainB.shape[0]-1),columns=['resultin'])

resultA=pd.concat([train,resultinA],axis=1,join='inner')
resultB=pd.concat([train,resultinB],axis=1,join='inner')
result=pd.concat([resultA,resultB],axis=0)
result=result.sort_index()

print result.head(),'\n'
print result.describe()
plt.figure(facecolor='white')
result['resultin'].plot(color='blue', label='Predict')
result['cnt'].plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('MSE: %.4f' % mean_squared_error(result['resultin'],result['cnt']))
plt.show()



