#-*- coding: utf-8 -*-
#阿里天池预测项目
import time
import datetime as dt
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
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import AdaBoostClassifier  


# 1数据提取
df = pd.read_table('../tmp/sample_tmp.txt',index_col='d')
df.index = pd.to_datetime(df.index)
df = df[['date','day_of_week','ifworkday','cnt']]
df['year']=df.index.year
df['month']=df.index.month
df['day']=df.index.day
# print df.info(),df.index,df.head()

# 2数据预处理
#训练数据
train=df[(df.index>='2013-01-01')&(df.index<'2015-01-01')]
x_train=train.loc[:,['day_of_week','ifworkday','year','month','day']].values
y_train=train.loc[:,'cnt'].values
#测试数据
original=df[(df.index>='2015-01-01')&(df.index<='2015-12-31')]
x_test=original.loc[:,['day_of_week','ifworkday','year','month','day']].values
y_test=original.loc[:,'cnt'].values

# 3数据建模
# model=linear_model.LinearRegression()
# model=DecisionTreeClassifier(max_depth=5)  
# model=RandomForestClassifier()
model=GradientBoostingRegressor(n_estimators=100)  
# model=GradientBoostingClassifier(n_estimators=100)  
# model=AdaBoostClassifier(model1,n_estimators=100)  

model.fit(x_train,y_train)    
target_in=model.predict(x_train)  
target_out=model.predict(x_test)

# 4数据模型评估
# 样本内评估
y_train=pd.DataFrame(y_train,columns=['cnt'])
target_in=pd.DataFrame(target_in,columns=['target'])
rusult_in=pd.concat([y_train,target_in],axis=1)
rusult_in['cnt'].plot(color='blue', label='Original')
rusult_in['target'].plot(color='red', label='Predict')
plt.legend(loc='best')
plt.title('IN-MSE: %.4f' % mean_squared_error(rusult_in['cnt'],rusult_in['target']))
plt.show()
# 样本外评估
y_test=pd.DataFrame(y_test,columns=['cnt'])
target_out=pd.DataFrame(target_out,columns=['target'])
rusult_out=pd.concat([y_test,target_out],axis=1)
rusult_out['cnt'].plot(color='blue', label='Original')
rusult_out['target'].plot(color='red', label='Predict')
plt.legend(loc='best')
plt.title('OUT-MSE: %.4f' % mean_squared_error(rusult_out['cnt'],rusult_out['target']))
plt.show()

