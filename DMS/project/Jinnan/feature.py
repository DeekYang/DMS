#-*- coding: utf-8 -*-
from mining import *
from sklearn.externals import joblib
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
	data = data.copy().fillna(0).astype('int64')
	# print(data['A10'].value_counts(dropna=False))

	# for i in data.columns:
	# 	group=data[i].value_counts(dropna=False)[data[i].value_counts(dropna=False)/data.shape[0]>0.01].index
	# 	print(data.loc[~data[i].isin(group),i])
	# 	data.loc[~data[i].isin(group),i] = 0

	data.loc[~data['A19'].isin([200,300]),'A19'] = 0	
	data.loc[~data['B1'].isin([320,300,350,340,370,290,310,330]),'B1'] = -1
	data.loc[~data['B12'].isin([800,1200]),'B12'] = 0
	data.loc[~data['B14'].isin([400,420,440]),'B14'] = 0

	data.loc[~data['A6'].isin([29,30,21,24,27,38,28]),'A6'] = 0
	data.loc[~data['A10'].isin([100,102,101,103]),'A10'] = 0
	data.loc[~data['A12'].isin([103,102,104,101]),'A12'] = 0
	data.loc[~data['A15'].isin([104,103,105,102]),'A15'] = 0
	data.loc[~data['A17'].isin([105,104,106,102,107,103]),'A17'] = 0
	data.loc[~data['A22'].isin([9,10]),'A22'] = 0
	data.loc[~data['A25'].isin([80,70,78,79]),'A25'] = 0
	data.loc[~data['A27'].isin([73,78,75,72,76,70,77]),'A27'] = 0
	data.loc[~data['B6'].isin([80,65,60]),'B6'] = 0
	data.loc[~data['B8'].isin([45,40,50]),'B8'] = 0

	data.loc[~data['A20'].isin([30,60]),'A20'] = 0
	data.loc[~data['A28'].isin([30,60]),'A28'] = 0
	data.loc[~data['B4'].isin([60,90,120]),'B4'] = 0
	data.loc[~data['B9'].isin([90,240,60,120,180,80]),'B9'] = 0
	data.loc[~data['B10'].isin([90,0,60,120,80,70]),'B10'] = -1
	data.loc[~data['B11'].isin([90,60,0]),'B11'] = -1
	
	# print(data['B6'].value_counts(dropna=False))

	return data

def get_inside_feature(data):
	data =data.copy()
	# data['inside_sampleid'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))
	data['inside_sum_num'] = data['A19']+data['B1']+data['B12']+data['B14']
	data['inside_sum_time'] = data['A20']+data['A28']+data['B4']+data['B9']+data['B10']+data['B11']
	inside_features = ['inside_sum_num','inside_sum_time']
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
		order_mean = sourse.groupby(i)['收率'].mean()
		order_max = sourse.groupby(i)['收率'].max()
		order_min = sourse.groupby(i)['收率'].min()
		order_std = sourse.groupby(i)['收率'].std().fillna(0)
		order_sum = sourse.groupby(i)['收率'].sum()
		order_median = sourse.groupby(i)['收率'].median()

		# data['new_'+i+'_mean'] = data[i].map(order_mean)
		# data['new_'+i+'_max'] = data[i].map(order_max)
		# data['new_'+i+'_min'] = data[i].map(order_min)
		# data['new_'+i+'_std'] = data[i].map(order_std)
		# data['new_'+i+'_sum'] = data[i].map(order_sum)
		# data['new_'+i+'_median'] = data[i].map(order_median)
		
		# for j in new_feature:
		# 	order_label = sourse.groupby(i)[j].count()
		# 	data['new_'+i+'_'+str(j)] = data[i].map(order_label)
	data = data.iloc[:,len_first:].copy()

	return data

def data_preprocess(data,flag):
	# 删除特征
	data.drop(['A1','A2','A3','A4','A8','A13','A18','A21','A23','B2','B3','B13',
	'A5','A7','A9','A11','A14','A16','A24','A26','B5','B7'],axis=1,inplace=True)

	# print(data.info())
	# print(data['A25'].head())
	# 非数字型特征清洗&转换为数字型特征	
	not_num_features = ['A20','A28','B4','B9','B10','B11']
	data[not_num_features] = not_num_clean(data[not_num_features])

	for i in not_num_features:
		data[i] = data[i].apply(not_num_transformation)
	
	# 数字型特征清洗
	num_features = ['A19','B1','B12','B14',
	'A6','A10','A12','A15','A17','A22','A25','A27','B6','B8']+not_num_features
	data[num_features] = num_clean(data[num_features])
	# print(data.info())
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

	out_first_features = [i for i in num_features if i not in inside_features]
	out = get_outside_feature(data[out_first_features])
	
	out_features = [i for i in out.columns]	
	data[out_features] = out
	
	num_features = num_features+out_features
	
	# 离散连续特征类型切分
	discrete_feature = [i for i in num_features if i not in inside_features+out_features]
	continuous_feature = [i for i in num_features if i not in discrete_feature]

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

def baseline_score(data):
	df_train = data
	X,Y=df_train.iloc[:,:-1].copy(),df_train['收率'].copy()

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, shuffle = False)
	clf = GradientBoostingRegressor()
	clf.fit(X_train, Y_train)
	Y_predict = clf.predict(X_test).round(3)
	best_score = metrics.mean_squared_error(Y_test, Y_predict)*0.5
	print("First_score: {:<8.8f}".format(best_score))
	'''
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

def new():
	model = joblib.load('../../data/project/testJinnan/temp/filename.pkl')
	i=0
	best = 0
	best_df = pd.DataFrame()
	for A19 in [200,300]:
		for B1 in [320,300,350,340,370,290,310,330]:
			for B12 in [800,1200]:
				for B14 in [400,420,440]:
					print(i)
					i+=1
					df_new=pd.DataFrame({
										'A19':A19,'B1':B1,'B12':B12,'B14':B14,
										# 'A6':A6,'A10':A10,'A12':A12,'A15':A15,'A17':A17,
										# 'A22':A22,'A25':A25,'A27':A27,'B6':B6,'B8':B8,
										'A6':29,'A10':100,'A12':103,'A15':104,'A17':105,
										'A22':9,'A25':80,'A27':73,'B6':80,'B8':45,


										'A20':'11:00-11:30','A28':'11:00-11:30','B4':'11:00-12:00',
										'B9':'11:00-12:30','B10':'11:00-12:30','B11':'11:00-12:30',
										'样本id':'sample_1750',
										'A1':0,'A2':0,'A3':0,'A4':0,'A8':0,'A13':0,
										'A18':0,'A21':0,'A23':0,'B2':0,'B3':0,'B13':0,
										'A5':0,'A7':0,'A9':0,'A11':0,'A14':0,'A16':0,
										'A24':0,'A26':0,'B5':0,'B7':0
									},index=list(range(1)))


					df_new_af = data_preprocess(df_new.reset_index(drop=True),flag='test')
					df_new_af['target'] = model.predict(df_new_af.iloc[:,:-1]).round(3)
					if df_new_af.loc[0,'target']>best:
						best = df_new_af.loc[0,'target']
						best_df = df_new
						print(best)
						
	print(best)
	print(best_df)
	best_df.to_csv('../../data/project/testJinnan/OUT/'+'optimize'+'.'+'csv',header=True,index=False)

if __name__ == "__main__":
	# 数据读取
	df_train = pd.read_csv('../../data/project/testJinnan/In/jinnan_round1_train_20181227.csv',encoding='gb18030')
	df_test_A = pd.read_csv('../../data/project/testJinnan/In/jinnan_round1_testA_20181227.csv',encoding='gb18030')
	df_test_B = pd.read_csv('../../data/project/testJinnan/In/jinnan_round1_testB_20190121.csv',encoding='gb18030')

	# 训练数据筛选
	df_train = df_train[(df_train['收率']>0.80)&(df_train['收率']<1)]
	df_train = df_train[~df_train['A25'].str.contains('1900')]
	df_train['A25'] = df_train['A25'].astype('int')

	# ******生成训练测试数据******
	df_train = data_preprocess(df_train.reset_index(drop=True),flag='train')
	df_test_A = data_preprocess(df_test_A.reset_index(drop=True),flag='test')
	df_test_B = data_preprocess(df_test_B.reset_index(drop=True),flag='test')	
	# print(df_train.shape,df_test_A.shape,df_test_B.shape)

	baseline_score(df_train)
		
	# # 输出临时数据集	
	df_train.to_csv('../../data/project/testJinnan/temp/'+'data_train'+'.'+'csv',header=True,index=False)
	df_test_A.to_csv('../../data/project/testJinnan/temp/'+'data_test_A'+'.'+'csv',header=True,index=False)
	df_test_B.to_csv('../../data/project/testJinnan/temp/'+'data_test_B'+'.'+'csv',header=True,index=False)

	# new()