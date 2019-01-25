#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def not_num_clean(data):
	data = data.copy()
	data['A20'] = data['A20'].str.replace('0分','0')
	data['A20'] = data['A20'].str.replace(':-','-')
	data['B4'] = data['B4'].str.replace(':-',':00-')
	data['B4'] = data['B4'].str.replace('1600','16:00')
	for i in data.columns:
		data[i] = data[i].fillna('0:00-0:00')
		data[i] = data[i].str.replace('；|;|"|:{2}',':')
	
	return data

def not_num_transformation(row):
	time = row.split('-')
	begin = time[0].split(':')
	beginT = int(begin[0])*3600+int(begin[1])*60
	end = time[1].split(':')
	endT = int(end[0])*3600+int(end[1])*60
	if endT<beginT:
		time_sum = (86400+(endT-beginT))/60
	else:
		time_sum = (endT-beginT)/60
	
	return time_sum

def num_clean(data):
	data = data.copy()

	data.loc[(data['A19']!=200)&(data['A19']!=300),'A19'] = data['A19'].value_counts(dropna=False).index[0]
	data.loc[(data['B1']!=320)&(data['B1']!=300)&(data['B1']!=350)
	&(data['B1']!=340)&(data['B1']!=370)&(data['B1']!=290)&(data['B1']!=310)
	&(data['B1']!=330)&(data['B1']!=380)&(data['B1']!=360)&(data['B1']!=280)
	&(data['B1']!=1200),'B1'] = data['B1'].value_counts(dropna=False).index[0]
	data.loc[(data['B12']!=800)&(data['B12']!=1200),'B12'] = data['B12'].value_counts(dropna=False).index[0]
	data.loc[(data['B14']!=400)&(data['B14']!=420)&(data['B14']!=440),'B14'] = data['B14'].value_counts(dropna=False).index[0]
	
	data.loc[(data['A22']!=9)&(data['A22']!=10),'A22'] = data['A22'].value_counts(dropna=False).index[0]
	data.loc[:,['A12','A15','A17']] = data[['A12','A15','A17']].round(0)
	data.loc[(data['A25']!=80)&(data['A25']!=70)&(data['A25']!=78)&(data['A25']!=79),'A25'] = data['A25'].value_counts(dropna=False).index[0]
	data.loc[(data['A27']!=73)&(data['A27']!=78)&(data['A27']!=75)&(data['A27']!=72)&(data['A27']!=76)&(data['A27']!=70)&(data['A27']!=77),'A27'] = data['A27'].value_counts(dropna=False).index[0]
	data.loc[(data['B8']!=45)&(data['B8']!=40)&(data['B8']!=50),'B8'] = data['B8'].value_counts(dropna=False).index[0]
	
	data.loc[(data['A20']!=30)&(data['A20']!=60),'A20'] = data['A20'].value_counts(dropna=False).index[0]
	data.loc[(data['A28']!=30)&(data['A28']!=60),'A28'] = data['A28'].value_counts(dropna=False).index[0]
	data.loc[(data['B4']!=60)&(data['B4']!=90)&(data['B4']!=120),'B4'] = data['B4'].value_counts(dropna=False).index[0]
	data.loc[(data['B9']!=90)&(data['B9']!=240)&(data['B9']!=60)&(data['B9']!=120)&(data['B9']!=180)&(data['B9']!=80),'B9'] = data['B9'].value_counts(dropna=False).index[0]
	data.loc[(data['B10']!=90)&(data['B10']!=0)&(data['B10']!=60)&(data['B10']!=120)&(data['B10']!=80)&(data['B10']!=70),'B10'] = data['B10'].value_counts(dropna=False).index[0]
	data.loc[(data['B11']!=90)&(data['B11']!=60)&(data['B11']!=0),'B11'] = data['B11'].value_counts(dropna=False).index[0]

	return data

def get_inside_feature(data):
	data =data.copy()
	data['inside_sampleid'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))
	data['inside_sum_num'] = data['A19']+data['B1']+data['B12']+data['B14']
	data['inside_sum_time'] = data['A20']+data['A28']+data['B4']+data['B9']+data['B10']+data['B11']
	inside_features = ['inside_sampleid','inside_sum_num','inside_sum_time']
	data = data[inside_features]
	return data

def get_outside_feature(data):
	est = preprocessing.KBinsDiscretizer(n_bins = 5, encode = 'onehot-dense')
	new = est.fit_transform(sourse[['收率']])
	new = pd.DataFrame(data=new)
	new_feature = [i for i in new.columns]
	sourse[new_feature]=new

	data = data.copy()
	len_first = len(data.columns)
	for i in data.columns:
		for j in new_feature:
			order_label = sourse.groupby(i)[j].count()
			data['new_'+i+'_'+str(j)] = data[i].map(order_label)
	data = data.iloc[:,len_first:].copy()

	return data

def data_preprocess(data,flag):
	# 删除特征
	data.drop(['A1','A2','A3','A4','A8','A13','A18','A21','A23','B2','B3','B13',
	'A5','A7','A9','A11','A14','A16','A24','A26','B5','B7'],axis=1,inplace=True)

	# 非数字型特征清洗&转换为数字型特征	
	not_num_features = ['A20','A28','B4','B9','B10','B11']
	data[not_num_features] = not_num_clean(data[not_num_features])

	for i in not_num_features:
		data[i] = data[i].apply(not_num_transformation)
	
	# 数字型特征清洗
	num_features = ['A19','B1','B12','B14',
	'A6','A10','A12','A15','A17','A22','A25','A27','B6','B8']+not_num_features
	data[num_features] = num_clean(data[num_features])

	# 内部特征构造
	inside_first_features = [i for i in num_features+['样本id']]
	inside = get_inside_feature(data[inside_first_features])
	
	inside_features = [i for i in inside.columns]	
	data[inside_features] = inside
	
	num_features = num_features+inside_features

	# 外部特征构造
	if flag == 'train':
		global sourse
		sourse = data[num_features+['收率']].copy()

	out_first_features = [i for i in num_features]
	out = get_outside_feature(data[out_first_features])
	
	out_features = [i for i in out.columns]	
	data[out_features] = out
	
	num_features = num_features+out_features
	
	# 离散连续特征类型切分
	discrete_feature = [i for i in num_features if i not in inside_features+out_features]
	continuous_feature = [i for i in num_features if i not in discrete_feature+out_features]

	# 离散型特征转换
	if flag == 'train' :
		global le_change
		le_change = data[discrete_feature].copy()
	
	le = preprocessing.LabelEncoder()
	for i in data[discrete_feature].columns:
		le.fit(le_change[i])		
		data.loc[:,i] = le.transform(data[i])

	if flag == 'train' :
		global ohe_change
		ohe_change = data[discrete_feature].copy()
		
	ohe = preprocessing.OneHotEncoder(categories = 'auto')
	ohe.fit(ohe_change)
	OneHot_data = ohe.transform(data[discrete_feature])
	OneHot_data = pd.DataFrame(data = OneHot_data.toarray())
	discrete_feature = [i for i in OneHot_data.columns]
	data[discrete_feature]=OneHot_data

	# 连续型特征转换
	# nml = preprocessing.MinMaxScaler()
	# data[continuous_feature] = nml.fit_transform(data[continuous_feature])

	# 特征选取
	if flag == 'train':
		data=data[discrete_feature+continuous_feature+['收率']]
	else:
		data=data[discrete_feature+continuous_feature+['样本id']]

	# print('离散型特征数：{} 连续型特征数：{}'.format(len(discrete_feature),len(continuous_feature)))

	return data

if __name__ == "__main__":
	# 数据读取
	df_train = pd.read_csv('../data/testJinnan/In/jinnan_round1_train_20181227.csv',encoding='gb18030')
	# df_test_A = pd.read_csv('../data/testJinnan/In/jinnan_round1_testA_20181227.csv',encoding='gb18030')
	df_test_B = pd.read_csv('../data/testJinnan/In/jinnan_round1_testB_20190121.csv',encoding='gb18030')

	# 训练数据筛选
	df_train = df_train[(df_train['收率']>0.80 )&(df_train['收率']<1)]
	df_train = df_train[~df_train['A25'].str.contains('1900')]
	df_train['A25'] = df_train['A25'].astype('int')

	# ******生成训练测试数据******
	df_train = data_preprocess(df_train.reset_index(drop=True),flag='train')
	# df_test_A = data_preprocess(df_test_A.reset_index(drop=True),flag='test')
	df_test_B = data_preprocess(df_test_B.reset_index(drop=True),flag='test')
	
	print(df_train.shape)
	# print(df_test_A.shape)
	print(df_test_B.shape)
	
	
	X,Y=df_train.iloc[:,:-1].copy(),df_train['收率'].copy()

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, shuffle = False)
	clf = GradientBoostingRegressor(n_estimators = 200)
	clf.fit(X_train, Y_train)
	Y_predict = clf.predict(X_test).round(3)
	best_score = metrics.mean_squared_error(Y_test, Y_predict)*0.5
	print("First_score: {:<8.8f}".format(best_score))
	init_cols = X.columns.tolist()[:].copy()
	best_cols = X.columns.tolist()[:].copy()
		
	for col in init_cols:
		best_cols.remove(col)
		X_train, X_test, Y_train, Y_test = train_test_split(X[best_cols], Y, test_size = 0.33, shuffle = False)
		clf = GradientBoostingRegressor(n_estimators=200)
		clf.fit(X_train, Y_train)
		Y_predict = clf.predict(X_test).round(3)
		score = metrics.mean_squared_error(Y_test, Y_predict)*0.5
		print("{} score: {:<8.8f}".format(col,score))
		if best_score-score > 0.0000002 :
			print("删除列名 {} 后 score: {:<8.8f}".format(col,score))
			best_score = score
		else :
			best_cols.append(col)
	print("Best_score: {:<8.8f}".format(best_score))
	
	'''
	# 输出临时数据集	
	# df_train = df_train[best_cols].copy()
	# df_test_B = df_test_B[best_cols].copy()
	# df_train.to_csv('../data/testJinnan/temp/'+'data_train'+'.'+'csv',header=True,index=False)
	# df_test_B.to_csv('../data/testJinnan/temp/'+'data_test'+'.'+'csv',header=True,index=False)
	'''