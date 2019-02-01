#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
from datetime import date,datetime
import multiprocessing,os,time

def label(row):
	if row['Date'] != -1:
		td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
		if td <= pd.Timedelta(15, 'D'):
			return 1
	return 0

def getDiscountType(row):
	if pd.isnull(row):
		return -1
	elif ':' in row:
		return 1
	else:
		return 2

def getDiscountRate(row):
	if pd.isnull(row):
		return 1
	elif ':' in row:
		rate = row.split(':')
		return 1.0-float(rate[1])/float(rate[0])
	else:
		return row

def getDiscountMan(row):
	if pd.isnull(row):
		return -1
	elif ':' in row:
		rate = row.split(':')
		return rate[0]
	else:
		return -1

def getDiscountJian(row):
	if pd.isnull(row):
		return -1
	elif ':' in row:
		rate = row.split(':')
		return rate[1]
	else:
		return -1

def getDistance(row):
	if pd.isnull(row):
		return -1
	else:
		return row

def getWeekday(row):
	# 获取星期几
	return row.weekday()+1

def getDaytype(row):
	if (row == 6) | (row == 7):
		return 1
	else:
		return 0

# 构造表内新特征
def get_feature_self(data):
	print("Run get_feature_self task %s" %(os.getpid()))
	start = time.time()
	# 构造特征
	data['Dis_type'] = data['Discount_rate'].apply(getDiscountType)
	data['Dis_rate'] = data['Discount_rate'].apply(getDiscountRate)
	data['Dis_man']  = data['Discount_rate'].apply(getDiscountMan)
	data['Dis_jian'] = data['Discount_rate'].apply(getDiscountJian)
	data['Distance'] = data['Distance'].apply(getDistance)
	data['Weekday']=data['Date_received'].apply(getWeekday)
	data['Day_type'] = data['Weekday'].apply(getDaytype)

	end = time.time()
	print('get_feature_self Task runs %0.2f seconds.' %(end - start))
	# print(data.columns.tolist())
	return data

# 构造统计汇总新特征--user
def userFeature(data):
	print("Run userFeature task -%s" %(os.getpid()))
	start = time.time()
	u = data[['User_id']].copy().drop_duplicates()
	
	# u_coupon_count : num of coupon received by user
	u1 = data[pd.notnull(data['Date_received'])][['User_id']].copy()
	u1['u_coupon_count'] = 1
	u1 = u1.groupby(['User_id'], as_index = False).count()


	# u_buy_count : times of user buy offline (with or without coupon)
	u2 = data[pd.notnull(data['Date'])][['User_id']].copy()
	u2['u_buy_count'] = 1
	u2 = u2.groupby(['User_id'], as_index = False).count()

	# u_buy_with_coupon : times of user buy offline (with coupon)
	u3 = data[pd.notnull(data['Date']) & pd.notnull(data['Date_received'])][['User_id']].copy()
	u3['u_buy_with_coupon'] = 1
	u3 = u3.groupby(['User_id'], as_index = False).count()

	# u_merchant_count : num of merchant user bought from
	u4 = data[pd.notnull(data['Date'])][['User_id', 'Merchant_id']].copy()
	u4.drop_duplicates(inplace = True)
	u4 = u4.groupby(['User_id'], as_index = False).count()
	u4.rename(columns = {'Merchant_id':'u_merchant_count'}, inplace = True)

	# u_min_distance
	utmp = data[pd.notnull(data['Date']) & pd.notnull(data['Date_received'])][['User_id', 'Distance']].copy()
	utmp.replace(-1, np.nan, inplace = True)
	u5 = utmp.groupby(['User_id'], as_index = False).min()
	u5.rename(columns = {'Distance':'u_min_distance'}, inplace = True)
	u6 = utmp.groupby(['User_id'], as_index = False).max()
	u6.rename(columns = {'Distance':'u_max_distance'}, inplace = True)
	u7 = utmp.groupby(['User_id'], as_index = False).mean()
	u7.rename(columns = {'Distance':'u_mean_distance'}, inplace = True)
	u8 = utmp.groupby(['User_id'], as_index = False).median()
	u8.rename(columns = {'Distance':'u_median_distance'}, inplace = True)

	user_feature = pd.merge(u, u1, on = 'User_id', how = 'left')
	user_feature = pd.merge(user_feature, u2, on = 'User_id', how = 'left')
	user_feature = pd.merge(user_feature, u3, on = 'User_id', how = 'left')
	user_feature = pd.merge(user_feature, u4, on = 'User_id', how = 'left')
	user_feature = pd.merge(user_feature, u5, on = 'User_id', how = 'left')
	user_feature = pd.merge(user_feature, u6, on = 'User_id', how = 'left')
	user_feature = pd.merge(user_feature, u7, on = 'User_id', how = 'left')
	user_feature = pd.merge(user_feature, u8, on = 'User_id', how = 'left')

	user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float')/user_feature['u_coupon_count'].astype('float')
	user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float')/user_feature['u_buy_count'].astype('float')
	
	user_feature = user_feature.fillna(0)

	end = time.time()
	print('userFeature Task runs %0.2f seconds.' %(end - start))

	return user_feature

# 提取特征
def get_feature(data):
	# 构造特征
	pool = multiprocessing.Pool(processes=4)

	feature_self = pool.apply_async(get_feature_self, (data,))
	user_feature_offline=pool.apply_async(userFeature, (feature_df_offline,))
	
	pool.close()
	pool.join()

	feature_self = feature_self.get()
	user_feature_offline = user_feature_offline.get()

	# 合并特征
	data = pd.merge(feature_self, user_feature_offline, how = 'left',on='User_id')
	
	# 选取特征
	# print(data.columns.tolist())
	feature=['User_id','Coupon_id','Date_received','Date',
	'Weekday','Day_type',
	'Dis_type','Dis_rate','Dis_man','Dis_jian',
	'u_coupon_count','u_buy_count','u_buy_with_coupon','u_merchant_count',
	'u_use_coupon_rate','u_buy_with_coupon_rate',
	'Distance','u_min_distance','u_max_distance','u_mean_distance','u_median_distance'
	]
	data=data[feature]

	# 处理空值
	# print(data.info())
	data=data.fillna(-1)

	return data

if __name__ == "__main__": 
	# 提取公共数据
	df_offline = readDatafile('../data/testO2O/In/ccf_offline_stage1_train_temp.csv')
	df_offline['Date_received'],df_offline['Date'] = pd.to_datetime(df_offline['Date_received']
		,format = '%Y-%m-%d'),pd.to_datetime(df_offline['Date'],format = '%Y-%m-%d')
	
	# 统计特征数据来源——train:(3.15-4.16); demo:(5.15-6.16)
	feature_df_offline = df_offline[((df_offline['Date']>'20160515') & (df_offline['Date']<'20160616')) 
	| ((pd.isnull(df_offline['Date'])) & ((df_offline['Date_received']>'20160515') & (df_offline['Date_received'] < '20160616')))].copy()
	# print(feature_df_offline.info())
	# print(feature_df_offline.head())

	# df_online = readDatafile('../data/testO2O/In/ccf_online_stage1_train_temp.csv')
	# df_online['Date_received'],df_online['Date'] = pd.to_datetime(df_online['Date_received']
	# 	,format = '%Y%m%d'),pd.to_datetime(df_online['Date'],format = '%Y%m%d')
	# feature_df_online = df_online[(df_online['Date'] < '20160516') | ((pd.isnull(df_online['Date'])) 
	# 	& (df_online['Date_received'] < '20160516'))].copy()

	# # 生成训练&测试数据
	train = df_offline[((df_offline['Date']>'20160501') & (df_offline['Date']<'20160531')) 
	| ((pd.isnull(df_offline['Date'])) & ((df_offline['Date_received']>'20160501') & (df_offline['Date_received'] < '20160531')))].copy().reset_index(drop=True)
	train = train[pd.notnull(train['Date_received'])].copy().reset_index(drop=True)
	# train=train.head(10)
	# train_data = get_feature(train)#注销开关
	# train_data['label'] = train_data.apply(label, axis = 1)#注销开关

	# 生成demo数据	
	demo = readDatafile('../data/testO2O/In/ccf_offline_stage1_test_revised_temp.csv')
	# demo = demo.head(10)
	demo['Date_received'] = pd.to_datetime(demo['Date_received'],format = '%Y%m%d')
	demo['Date'] = -1
	demo_data=get_feature(demo)
	
	# 输出临时数据集	
	# train_data.to_csv('../data/testO2O/temp/'+'data_train'+'.'+'csv',header=True,index=False)
	# demo_data.to_csv('../data/testO2O/temp/'+'data_demo'+'.'+'csv',header=True,index=False)