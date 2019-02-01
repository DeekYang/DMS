#-*- coding: utf-8 -*-
#阿里天池预测项目

import time
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

rusult_all=pd.DataFrame(columns=['date','brand','month','day','day_of_week','ifworkday','target'])
for x in range(10):
	#1数据提取
	workday = pd.read_excel('../data/workday.xlsx')
	train = pd.read_table('../data/all.txt')
	z_demo = pd.read_table('../data/fusai_test_B_20180227.txt') 

	#2数据预处理
	#训练数据
	train=train[train['brand']==x+1]
	train['d'],train['w']=dt.datetime(2012,12,30)+dt.timedelta(train.head(1)['date'].values[0].astype('int')//7*7),0
	train[['day_of_week','w']]=train[['day_of_week','w']].astype('int')
	train=train.reset_index(drop=True)
	for i in range(len(train)-1):
	    if (train.loc[i,'day_of_week']>=train.loc[i+1,'day_of_week']):
	        train.loc[i+1:,'w']=train.loc[i+1:,'w']+1
	for i in range(len(train)):        
	    train.loc[i,'d']=train.loc[i,'d']+ dt.timedelta(days=train.loc[i,'w']*7+train.loc[i,'day_of_week'])        
	train=pd.merge(train,workday,left_on=train['d'],right_on=workday['wdate'],how='left')
	train.set_index(['d'],inplace=True)
	train['year']=train.index.year
	train['month']=train.index.month
	train['day']=train.index.day
	train = train[['date','brand','year','month','day','day_of_week','ifworkday','cnt']]#过滤属性
	x_train=train.loc[:,['year','month','day','day_of_week','ifworkday']].values
	y_train=train.loc[:,'cnt'].values

	#实际数据
	z_demo=z_demo[z_demo['brand']==x+1]
	z_demo['d'],z_demo['w']=dt.datetime(2012,12,30)+dt.timedelta(z_demo.head(1)['date'].values[0].astype('int')//7*7),0
	z_demo[['day_of_week','w']]=z_demo[['day_of_week','w']].astype('int')
	z_demo=z_demo.reset_index(drop=True)
	for i in range(len(z_demo)-1):
	    if (z_demo.loc[i,'day_of_week']>=z_demo.loc[i+1,'day_of_week']):
	        z_demo.loc[i+1:,'w']=z_demo.loc[i+1:,'w']+1
	for i in range(len(z_demo)):
	    z_demo.loc[i,'d']=z_demo.loc[i,'d']+ dt.timedelta(days=z_demo.loc[i,'w']*7+z_demo.loc[i,'day_of_week'])
	z_demo=pd.merge(z_demo,workday,left_on=z_demo['d'],right_on=workday['wdate'],how='left')
	z_demo.set_index(['d'],inplace=True)
	z_demo['year']=z_demo.index.year
	z_demo['month']=z_demo.index.month
	z_demo['day']=z_demo.index.day
	z_demo = z_demo[['date','brand','year','month','day','day_of_week','ifworkday']]#过滤属性
	x_z_demo=z_demo.loc[:,['year','month','day','day_of_week','ifworkday']].values#预测输入数据

	#3数据建模
	model=GradientBoostingRegressor(n_estimators=120)  
	model.fit(x_train,y_train)    
	target_out=model.predict(x_z_demo)

	#4预测结果合并
	z_demo=z_demo.reset_index()
	target_out=pd.DataFrame(target_out,columns=['target'])
	rusult_out=pd.concat([z_demo,target_out],axis=1)
	rusult_out['target']=rusult_out['target'].round(0)
	rusult_out.set_index(['d'],inplace=True)# 模拟还原时间索引列

	rusult_all=rusult_all.append(rusult_out)

rusult_all=rusult_all[['date','brand','target']].astype('int')
rusult_all=rusult_all.sort_values(['date','brand'])


# 5数据运用输出
rusult_all.to_csv('../tmp/sample_A_'+time.strftime("%Y%m%d", time.localtime())+'.txt',sep='\t',header=False,index=False)

