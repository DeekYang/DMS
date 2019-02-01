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
df = df[['date','ifworkday','cnt']]
# print df.info(),df.index,df.head()

# 2数据预处理
train=df[(df.index>='2014-12-01')&(df.index<'2015-01-01')]#训练数据,取两年数据
original=df[(df.index>='2015-01-01')&(df.index<='2015-12-31')]#实际数据
# print train.shape,train.head()
# print original.shape,original.head()
trainlist=pd.concat([train,original],axis=0)
# 3数据建模
train=trainlist.iloc[0:train.shape[0]]
for i in range(1):
	model = ARMA(train['cnt'], order=(1,1)).fit( disp=-1)# ARMA_model
	result=pd.DataFrame(model.predict(start=0,end=train.shape[0]),columns=['resultin'])
	trainlist.iloc[train.shape[0]+i,2]=result.iloc[train.shape[0]].values
	train = trainlist.iloc[i+1:train.shape[0]+i+1]

# 4数据模型评估
# 4.1样本外评估
result=pd.merge(original,trainlist,on='date',left_index=True,how='left')

plt.figure(facecolor='white')
# result['cnt_x'].plot(color='blue', label='Original')
# result['cnt_y'].plot(color='red', label='Predict')
# plt.legend(loc='best')
# plt.title('MSE: %.4f' % mean_squared_error(result['cnt_x'],result['cnt_y']))
plt.show()

