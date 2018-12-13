#-*- coding: utf-8 -*-

import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *

from datetime import date

# # ××××××××××提取数据××××××××××
# offline = readDatafile('data/ccf_offline_stage1_train.csv').drop_duplicates()
# offline['Date_received'] = pd.to_datetime(offline['Date_received'],format = '%Y%m%d')
# offline['Date'] = pd.to_datetime(offline['Date'],format = '%Y%m%d')
# online = readDatafile('data/ccf_online_stage1_train.csv').drop_duplicates()
# online['Date_received'] = pd.to_datetime(online['Date_received'],format = '%Y%m%d')
# online['Date'] = pd.to_datetime(online['Date'],format = '%Y%m%d')
demo=readDatafile('data/ccf_offline_stage1_test_revised.csv').head(100)
demo['Date_received'] = pd.to_datetime(demo['Date_received'],format = '%Y%m%d')

print(demo.info())
print(demo.head())

'''
source_offline = offline[((offline['Date_received'] > '20160501') 
& (offline['Date_received'] < '20160601')) | pd.isnull(offline['Date_received'])].copy().head()
source_online = online[((online['Date_received'] > '20160501') 
& (online['Date_received'] < '20160601')) | pd.isnull(online['Date_received'])].copy().head()
data = offline[(offline['Date_received'] >= '20160601')].copy().head()

# ××××××××××提取特征××××××××××
# **********offline**********
u = source_offline[['User_id']].drop_duplicates()

u1 = source_offline[pd.isnull(source_offline['Date_received']) 
& pd.notnull(source_offline['Date'])][['User_id']]
u1['u_nocoupon_count'] = 1
u1 = u1.groupby(['User_id'], as_index = False).count()

u2 = source_offline[pd.notnull(source_offline['Date_received']) 
& ((source_offline['Date']-source_offline['Date_received']>pd.Timedelta(15, 'D'))
| (pd.isnull(source_offline['Date'])))][['User_id']]
u2['u_withcoupon_out15day_count'] = 1
u2 = u2.groupby(['User_id'], as_index = False).count()

u3 = source_offline[pd.notnull(source_offline['Date_received'])
& (source_offline['Date']-source_offline['Date_received'] <= pd.Timedelta(15, 'D'))][['User_id']]
u3['u_withcoupon_in15day_count'] = 1
u3 = u3.groupby(['User_id'], as_index = False).count()

feature_offline = pd.merge(u, u1, on = 'User_id', how = 'left')
feature_offline = pd.merge(feature_offline, u2, on = 'User_id', how = 'left')
feature_offline = pd.merge(feature_offline, u3, on = 'User_id', how = 'left')

# **********online**********
m = source_online[['User_id']].drop_duplicates()

m1 = source_online[pd.isnull(source_online['Date_received']) 
& pd.notnull(source_online['Date'])][['User_id']]
m1['m_nocoupon_count'] = 1
m1 = m1.groupby(['User_id'], as_index = False).count()

m2 = source_online[pd.notnull(source_online['Date_received']) 
& ((source_online['Date']-source_online['Date_received']>pd.Timedelta(15, 'D'))
| (pd.isnull(source_online['Date'])))][['User_id']]
m2['m_withcoupon_out15day_count'] = 1
m2 = m2.groupby(['User_id'], as_index = False).count()

m3 = source_online[pd.notnull(source_online['Date_received'])
& (source_online['Date']-source_online['Date_received'] <= pd.Timedelta(15, 'D'))][['User_id']]
m3['m_withcoupon_in15day_count'] = 1
m3 = m3.groupby(['User_id'], as_index = False).count()

feature_online = pd.merge(m, m1, on = 'User_id', how = 'left')
feature_online = pd.merge(feature_online, m2, on = 'User_id', how = 'left')
feature_online = pd.merge(feature_online, m3, on = 'User_id', how = 'left')

# **********合并特征**********
for i in data.index:
	if pd.isnull(data.loc[i,'Date_received']):
		data.loc[i,'label']=-1
	elif pd.to_datetime(data.loc[i,'Date'], format='%Y%m%d')-pd.to_datetime(data.loc[i,'Date_received'], format='%Y%m%d')<= pd.Timedelta(15, 'D'):
		data.loc[i,'label']=1
	else:
		data.loc[i,'label']=0
	
	
	if pd.isnull(data.loc[i,'Discount_rate']):
		data.loc[i,'Dis_type'] = -1
		data.loc[i,'Dis_rate'] = 1
		data.loc[i,'Dis_man'] = -1
		data.loc[i,'Dis_jian'] = -1
	elif ':' in data.loc[i,'Discount_rate']:
		data.loc[i,'Dis_type'] = 1
		rate=data.loc[i,'Discount_rate'].split(':')
		data.loc[i,'Dis_rate'] = 1.0-float(rate[1])/float(rate[0])
		data.loc[i,'Dis_man'] = float(rate[0])
		data.loc[i,'Dis_jian'] = float(rate[1])
	else:
		data.loc[i,'Dis_type'] = 2
		data.loc[i,'Dis_rate'] = data.loc[i,'Discount_rate']
		data.loc[i,'Dis_man'] = -1
		data.loc[i,'Dis_jian'] = -1
	
	if pd.isnull(data.loc[i,'Distance']):
		data.loc[i,'Distance']=-1

	data.loc[i,'weekday'] =	data.loc[i,'Date_received'].weekday()+1
	
	if (data.loc[i,'weekday'] == 6) | (data.loc[i,'weekday'] == 7):
		data.loc[i,'day_type'] = 1
	else:
		data.loc[i,'day_type'] = 0 	
	# print(data.loc[i,['Date_received','weekday','day_type']])

data = pd.merge(data, feature_offline, on = 'User_id', how = 'left')
data = pd.merge(data, feature_online, on = 'User_id', how = 'left')

# **********处理demo数据**********
for j in demo.index:
	if pd.isnull(demo.loc[i,'Discount_rate']):
		demo.loc[i,'Dis_type'] = -1
		demo.loc[i,'Dis_rate'] = 1
		demo.loc[i,'Dis_man'] = -1
		demo.loc[i,'Dis_jian'] = -1
	elif ':' in demo.loc[i,'Discount_rate']:
		demo.loc[i,'Dis_type'] = 1
		rate=demo.loc[i,'Discount_rate'].split(':')
		demo.loc[i,'Dis_rate'] = 1.0-float(rate[1])/float(rate[0])
		demo.loc[i,'Dis_man'] = float(rate[0])
		demo.loc[i,'Dis_jian'] = float(rate[1])
	else:
		demo.loc[i,'Dis_type'] = 2
		demo.loc[i,'Dis_rate'] = demo.loc[i,'Discount_rate']
		demo.loc[i,'Dis_man'] = -1
		demo.loc[i,'Dis_jian'] = -1
	
	if pd.isnull(demo.loc[i,'Distance']):
		demo.loc[i,'Distance']=-1

	demo.loc[i,'weekday'] =	demo.loc[i,'Date_received'].weekday()+1
	
	if (demo.loc[i,'weekday'] == 6) | (demo.loc[i,'weekday'] == 7):
		demo.loc[i,'day_type'] = 1
	else:
		demo.loc[i,'day_type'] = 0 	
demo = pd.merge(demo, feature_offline, on = 'User_id', how = 'left')
demo = pd.merge(demo, feature_online, on = 'User_id', how = 'left')

# ××××××××××输出临时数据集××××××××××
# print(data.info())
# print(demo.info())
# print(demo.head())
# data.to_csv('temp/'+'data'+'.'+'csv',header=True,index=False)
# demo.to_csv('temp/'+'data_test'+'.'+'csv',header=True,index=False)

'''