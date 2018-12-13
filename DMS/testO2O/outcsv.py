#-*- coding: utf-8 -*-

import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
from datetime import datetime

from sklearn.externals import joblib

# 提取测试数据&&模型
dftest = readDatafile('temp/data_test.csv')
clf = joblib.load('temp/filename.pkl')

# 运用输出
dftest = dftest.fillna(-1)
feature=[ 'Dis_type','Dis_rate','Dis_man','Dis_jian','Distance',
'u_nocoupon_count','u_withcoupon_out15day_count','u_withcoupon_in15day_count',
'm_nocoupon_count','m_withcoupon_out15day_count','m_withcoupon_in15day_count','weekday','day_type']

pred = clf.predict_proba(dftest[feature].as_matrix())
dftest['Probability'] = pred[:,1]

dftest=dftest[['User_id','Coupon_id','Date_received','Probability']]
print(dftest.info())

print(type((dftest.loc[1,'Date_received'])))
print(dftest.loc[1,'Date_received'].split('-'))
# dftest['Date_received']=pd.to_datetime(dftest['Date_received'],format = '%Y%m%d')
# dftest['Date_received'].to_string()
# print(dftest['Date_received'].astype(str))

# print (dftest.info())
# dftest.to_csv('out/'+'submit.csv', index=False, header=False)
'''


# test = readDatafile('out/sample_submission.csv')
test2 = readDatafile('out/submit.csv')

# print (test.info(),test2.info())

print(test2.head())

date = datetime.now()
detester = int(date.strftime('%Y%m%d'))
print(detester)
print(type(detester))
'''