#-*- coding: utf-8 -*-
#阿里天池预测项目

import data_exploration as exp
import time
import datetime as dt
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as ACF
from statsmodels.tsa.stattools import pacf as PACF
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import isnan

# 1数据提取
df = pd.read_table('../data/train_20171215.txt')
workday=pd.read_excel('../data/holiday.xlsx')

# 2数据预处理
# 2.1数据集成
df = df[['date','day_of_week','cnt']]
df = df.groupby(['date','day_of_week'],as_index=False).sum()
df['d'],df['w']=dt.datetime(2012,12,30),0
# 2.2数据转换
df[['day_of_week','w']]=df[['day_of_week','w']].astype('int')
df['cnt']=df['cnt'].astype('float64')
for i in range(len(df)-1):
    if (df.loc[i,'day_of_week']>=df.loc[i+1,'day_of_week']):
        df.loc[i+1:,'w']=df.loc[i+1:,'w']+1
    df.loc[i,'d']=df.loc[i,'d']+ dt.timedelta(days=df.loc[i,'w']*7+df.loc[i,'day_of_week'])    
df=pd.merge(df,workday,left_on=df['d'],right_on=workday['dateY'],how='left')
df.set_index(['d'],inplace=True)# 模拟还原时间索引列

# print df.head(10)
# from statsmodels.tsa.seasonal import seasonal_decompose
# decomposition = seasonal_decompose(timeSeries, model="additive")
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid
# timeSeries=timeSeries.asfreq(freq='D',method='bfill')#重置索引频率
# 2.3数据清洗
# df15 = df[(df['day_of_week'] >= 1)&(df['day_of_week'] <= 5)][['date','day_of_week','cnt','ifworkday']]
df15 = df[df['ifworkday'] == 1][['date','day_of_week','cnt','ifworkday']]
df15['cntcopy']=df15['cnt']
min=df15['cnt'].quantile(0.25)-(df15['cnt'].quantile(0.75)-df15['cnt'].quantile(0.25))*1
max=df15['cnt'].quantile(0.75)+(df15['cnt'].quantile(0.75)-df15['cnt'].quantile(0.25))*1
for j in df15.index:
    if (df15['cntcopy'][j]<min or df15['cntcopy'][j]>max):
        df15.loc[j,'cntcopy']=0
avr=round(df15['cnt'].mean(),2)
for j in range(len(df15)):
    if (df15['cnt'][j]<min or df15['cnt'][j]>max):
        if isnan(df15['cnt'].rolling(window=3,center=False).mean()[j]):
            df15['cnt']=df15['cnt'].replace(to_replace=df15['cnt'][j],value=avr)
        else:
            df15['cnt']=df15['cnt'].replace(to_replace=df15['cnt'][j],value=round(df15['cnt'].rolling(window=14,center=False).mean()[j]))
df15=df15[['date','day_of_week','cnt','ifworkday']]

# df67 = df[(df['day_of_week'] >= 6)&(df['day_of_week'] <= 7)][['date','day_of_week','cnt','ifworkday']]
df67 = df[df['ifworkday'] == 0][['date','day_of_week','cnt','ifworkday']]
df67['cntcopy']=df67['cnt']
min=df67['cnt'].quantile(0.25)-(df67['cnt'].quantile(0.75)-df67['cnt'].quantile(0.25))*1
max=df67['cnt'].quantile(0.75)+(df67['cnt'].quantile(0.75)-df67['cnt'].quantile(0.25))*1
for j in df67.index:
    if (df67['cntcopy'][j]<min or df67['cntcopy'][j]>max):
        df67.loc[j,'cntcopy']=0
avr=round(df67['cnt'].mean(),2)
for j in range(len(df67)):
    if (df67['cnt'][j]<min or df67['cnt'][j]>max):
        # print df67['cnt'][j],df67.index[j]
        if isnan(df67['cnt'].rolling(window=7,center=False).mean()[j]):
            df67['cnt']=df67['cnt'].replace(to_replace=df67['cnt'][j],value=avr)
        else:
            df67['cnt']=df67['cnt'].replace(to_replace=df67['cnt'][j],value=round(df67['cnt'].rolling(window=3,center=False).mean()[j]))
df67=df67[['date','day_of_week','cnt','ifworkday']]

# df=pd.concat([df15,df67],axis=0)
df=df.sort_values('date')

# 2.4数据规约
df=df.head(df.shape[0])
df.to_csv('../tmp/sample_tmp.txt',sep='\t')
# print df.head()

# 3数据探索
timeSeries=df[['day_of_week','cnt','ifworkday']]
# timeSeries=timeSeries[(timeSeries['ifworkday']!=1)]
print timeSeries.head()
print timeSeries.describe()
count = timeSeries.groupby(['day_of_week','ifworkday'],as_index=False).mean()
print count
timeSeries['cnt'].plot()
# timeSeries[timeSeries['day_of_week']==7]['cnt'].plot(label='7')
plt.legend(loc='best')
plt.show()


# exp.draw_ts(timeSeries['cnt'])
# exp.testStationarity(timeSeries['cnt'])
# exp.testLBQ(timeSeries['cnt'])
# exp.draw_acf_pacf(timeSeries['cnt'], lags=31)




