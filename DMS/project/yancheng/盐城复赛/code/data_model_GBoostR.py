#-*- coding: utf-8 -*-
#阿里天池预测项目
import time
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import AdaBoostClassifier  

sumin=0
sumout=0
for x in range(10):
	# 1数据提取
	df = pd.read_table('../data/fusai_train_20180227.txt')
	workday=pd.read_excel('../data/workday.xlsx')

	# 2数据预处理
	df=df[df['brand']==x+1]
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

	# 2数据预处理
	#训练数据
	train=df[(df.index>='2013-01-01')&(df.index<'2015-07-01')]
	# print train.groupby(['day']).sum()
	x_train=train.loc[:,['day_of_week','ifworkday','year','month','day']].values
	y_train=train.loc[:,'cnt'].values
	#测试数据
	original=df[(df.index>='2015-07-01')&(df.index<='2016-12-31')]
	x_test=original.loc[:,['day_of_week','ifworkday','year','month','day']].values
	y_test=original.loc[:,'cnt'].values

	# # 3数据建模
	# model=linear_model.LinearRegression()
	# model=DecisionTreeClassifier(max_depth=5)  
	# model=RandomForestClassifier()
	model=GradientBoostingRegressor(learning_rate=0.01, n_estimators=100,loss='huber')  
	# # model=GradientBoostingClassifier(n_estimators=100)  
	# # model=AdaBoostClassifier(model1,n_estimators=100)  

	model.fit(x_train,y_train)    
	target_in=model.predict(x_train)  
	target_out=model.predict(x_test)

	# # 4数据模型评估
	# # 样本内评估
	y_train_in=pd.DataFrame(y_train,columns=['cnt'])
	target_in=pd.DataFrame(target_in,columns=['target'])
	rusult_in=pd.concat([y_train_in,target_in],axis=1)
	# rusult_in['cnt'].plot(color='blue', label='Original')
	# rusult_in['target'].plot(color='red', label='Predict')
	# plt.legend(loc='best')
	# plt.title('IN-MSE: %.4f' % mean_squared_error(rusult_in['cnt'],rusult_in['target']))
	# plt.show()
	print mean_squared_error(rusult_in['cnt'],rusult_in['target'])
	sumin=sumout+mean_squared_error(rusult_in['cnt'],rusult_in['target'])
	# 样本外评估
	y_test=pd.DataFrame(y_test,columns=['cnt'])
	target_out=pd.DataFrame(target_out,columns=['target'])
	rusult_out=pd.concat([y_test,target_out],axis=1)
	# rusult_out['cnt'].plot(color='blue', label='Original')
	# rusult_out['target'].plot(color='red', label='Predict')
	# plt.legend(loc='best')
	# plt.title('OUT-MSE: %.4f' % mean_squared_error(rusult_out['cnt'],rusult_out['target']))
	# plt.show()
	print mean_squared_error(rusult_out['cnt'],rusult_out['target'])
	sumout=sumout+mean_squared_error(rusult_out['cnt'],rusult_out['target'])
print '\n',sumin/10
print '\n',sumout/10
