#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
from sklearn.externals import joblib

# 提取数据&&模型
dftest = readDatafile('../data/testO2O/temp/data_demo.csv')
clf = joblib.load('../data/testO2O/temp/filename.pkl')

# 选取特征
feature=['User_id','Coupon_id','Date_received',
'Weekday','Day_type',
'Dis_type','Dis_rate','Dis_man','Dis_jian',
'u_coupon_count','u_buy_count','u_buy_with_coupon','u_merchant_count',
'u_use_coupon_rate','u_buy_with_coupon_rate',
'Distance','u_min_distance','u_max_distance','u_mean_distance','u_median_distance'
]
dftest=dftest[feature]

# 运用输出
pred = clf.predict_proba(dftest.iloc[:,3:].as_matrix())
dftest['Probability'] = pred[:,1]
dftest=dftest[['User_id','Coupon_id','Date_received','Probability']].copy()
dftest.to_csv('../data/testO2O/Out/'+'submit.csv', index=False, header=False)