#-*- coding: utf-8 -*-
#阿里天池预测项目

import data_exploration as exp
import time
from datetime import datetime
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import isnan


# 1数据提取
df = pd.read_table('../data/fusai_train_20180227.txt')
workday=pd.read_excel('../data/workday.xlsx')

# 2数据预处理
df=df[df['brand']==9]#5 8 9有异常波动情况 
df['d'],df['w']=dt.datetime(2012,12,30)+dt.timedelta(df.head(1)['date'].values[0].astype('int')//7*7),0
df[['day_of_week','w']]=df[['day_of_week','w']].astype('int')
df['cnt']=df['cnt'].astype('float64')
df=df.reset_index(drop=True)
for i in range(len(df)-1):
	if (df.loc[i,'day_of_week']>=df.loc[i+1,'day_of_week']):
	    df.loc[i+1:,'w']=df.loc[i+1:,'w']+1
for i in range(len(df)):       
	df.loc[i,'d']=df.loc[i,'d']+ dt.timedelta(days=df.loc[i,'w']*7+df.loc[i,'day_of_week'])    
df=pd.merge(df,workday,left_on=df['d'],right_on=workday['wdate'],how='left')
df.set_index(['d'],inplace=True)
df=df[['brand','date','day_of_week','ifworkday','cnt']]
df['year']=df.index.year-2000
df['month']=df.index.month
df['day']=df.index.day
# 3数据探索
print df.corr()['cnt']
# print df.head(20),'\n',df.tail()
# print df.shape
# exp.draw_ts(df['cnt'])





