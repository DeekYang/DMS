#-*- coding: utf-8 -*-
#阿里天池预测项目

import time
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# 1数据提取
workday = pd.read_excel('../data/workday.xlsx')
train = pd.read_table('../data/train_20171215.txt')
x_demo = pd.read_table('../data/test_A_20171225.txt')
y_demo = pd.read_table('../data/answer_A_20180225.txt',header=-1)
y_demo = y_demo.rename(columns={0: "adate", 1: "cnt"})

z_demo = pd.read_table('../data/test_B_20171225.txt') 
# 2数据预处理
#训练数据
train = train[['date','day_of_week','cnt']]#删除汽车品牌属性
train = train.groupby(['date','day_of_week'],as_index=False).sum()
train['d'],train['w']=dt.datetime(2012,12,30),0
train[['day_of_week','w']]=train[['day_of_week','w']].astype('int')
for i in range(len(train)-1):
    if (train.loc[i,'day_of_week']>=train.loc[i+1,'day_of_week']):
        train.loc[i+1:,'w']=train.loc[i+1:,'w']+1
    train.loc[i,'d']=train.loc[i,'d']+ dt.timedelta(days=train.loc[i,'w']*7+train.loc[i,'day_of_week'])        
train=pd.merge(train,workday,left_on=train['d'],right_on=workday['wdate'],how='left')
train.set_index(['d'],inplace=True)
train['month']=train.index.month
train['day']=train.index.day
train = train[['date','month','day','day_of_week','ifworkday','cnt']]#过滤属性
train=train[(train.index>='2013-01-01')&(train.index<'2017-01-01')]
# print train.head(),'\n',train.tail(),train.shape
x_train=train.loc[:,['month','day','day_of_week','ifworkday']].values
y_train=train.loc[:,'cnt'].values

#实际数据
x_demo['d'],x_demo['w']=dt.datetime(2016,04,03),0
x_demo[['day_of_week','w']]=x_demo[['day_of_week','w']].astype('int')
for i in range(len(x_demo)-1):
    if (x_demo.loc[i,'day_of_week']>=x_demo.loc[i+1,'day_of_week']):
        x_demo.loc[i+1:,'w']=x_demo.loc[i+1:,'w']+1
    x_demo.loc[i,'d']=x_demo.loc[i,'d']+ dt.timedelta(days=x_demo.loc[i,'w']*7+x_demo.loc[i,'day_of_week'])
x_demo=pd.merge(x_demo,workday,left_on=x_demo['d'],right_on=workday['wdate'],how='left')
x_demo.set_index(['d'],inplace=True)
x_demo['month']=x_demo.index.month
x_demo['day']=x_demo.index.day
x_demo=pd.merge(x_demo,y_demo,left_on=x_demo['date'],right_on=y_demo['adate'],how='left')
x_demo.set_index(['wdate'],inplace=True)
x_demo = x_demo[['date','month','day','day_of_week','ifworkday','cnt']]#过滤属性
# print x_demo.head(),'\n',x_demo.tail(),x_demo.shape
xx_demo=x_demo.loc[:,['month','day','day_of_week','ifworkday']].values
yy_demo=x_demo.loc[:,'cnt'].values

# 2017年数据
z_train=pd.concat([train,x_demo],axis=0)
z_train=z_train.head(z_train.shape[0]-1)
# print z_train.head(),'\n',z_train.tail(),z_train.shape
x_z_train=z_train.loc[:,['month','day','day_of_week','ifworkday']].values#训练输入数据
y_z_train=z_train.loc[:,'cnt'].values#训练输出数据
# x_z_demo
z_demo['d'],z_demo['w']=dt.datetime(2017,02,12),0
z_demo[['day_of_week','w']]=z_demo[['day_of_week','w']].astype('int')
for i in range(len(z_demo)-1):
    if (z_demo.loc[i,'day_of_week']>=z_demo.loc[i+1,'day_of_week']):
        z_demo.loc[i+1:,'w']=z_demo.loc[i+1:,'w']+1
    z_demo.loc[i,'d']=z_demo.loc[i,'d']+ dt.timedelta(days=z_demo.loc[i,'w']*7+z_demo.loc[i,'day_of_week'])
z_demo=pd.merge(z_demo,workday,left_on=z_demo['d'],right_on=workday['wdate'],how='left')
z_demo.set_index(['d'],inplace=True)
z_demo['month']=z_demo.index.month
z_demo['day']=z_demo.index.day
z_demo = z_demo[['date','month','day','day_of_week','ifworkday']]#过滤属性
z_demo=z_demo.head(z_demo.shape[0]-1)
# print z_demo.head(),'\n',z_demo.tail(),z_demo.shape
x_z_demo=z_demo.loc[:,['month','day','day_of_week','ifworkday']].values#预测输入数据


# 3数据建模
model=GradientBoostingRegressor(n_estimators=120)  
model.fit(x_z_train,y_z_train)    
target_out=model.predict(x_z_demo)

# 4数据模型评估
z_demo=z_demo.reset_index()
# print z_demo.head(),'\n',z_demo.tail(),z_demo.shape
target_out=pd.DataFrame(target_out,columns=['target'])
# print target_out.head(),'\n',target_out.tail(),target_out.shape
rusult_out=pd.concat([z_demo,target_out],axis=1)
# print rusult_out.head(),'\n',rusult_out.tail(),rusult_out.shape
rusult_out.set_index(['d'],inplace=True)# 模拟还原时间索引列
rusult_out['target'].plot(color='red', label='Predict')
plt.legend(loc='best')
plt.show()
print rusult_out.head()
# 5数据运用输出
rusult_out['target']=rusult_out['target'].round(0)
rusult_out=rusult_out[['date','target','day_of_week','ifworkday']]
print rusult_out.head(),'\n',rusult_out.tail(),rusult_out.shape
rusult_out.to_csv('../tmp/sample_A_'+time.strftime("%Y%m%d", time.localtime())+'.txt',sep='\t',header=False,index=True)

