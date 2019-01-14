#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
from sklearn import preprocessing
from scipy.sparse import coo_matrix, hstack,vstack

def get_time(row):
	time = row.split('-')
	begin = time[0].split(':')
	beginT=int(begin[0])*3600+int(begin[1])*60
	end = time[1].split(':')
	endT=int(end[0])*3600+int(end[1])*60
	if endT<beginT:
		time_sum = (86400+(endT-beginT))/60
	else:
		time_sum = (endT-beginT)/60
	
	return time_sum

def get_outFeature(data):
	sourseData = sourse[['B14','收率']].copy()	#改
	est = preprocessing.KBinsDiscretizer(n_bins=5, encode='onehot-dense')
	new = est.fit_transform(sourseData[['收率']])
	new = pd.DataFrame(data=new)
	new_feature = [i for i in new.columns]
	sourseData[new_feature]=new

	data = data.copy()
	l=len(data.columns)
	for i in data.columns:
		for j in new_feature:
			order_label = sourseData.groupby(i)[j].mean()
			data['new_'+i+'_'+str(j)] = data[i].map(order_label)
	data = data.iloc[:,l:].copy()

	return data

# 数据预处理&&特征工程
def get_feature(data,flag):
	# 删除特征
	data.drop(['A5','A7','A8','A9','A11','A14','A16','A23','A24','A26','B5','B7'],axis=1,inplace=True)

	# 非数字型特征提取
	num_columns = ['A1','A2','A3','A4','A19','A21','A22','B1','B12','B14',
	'A6','A10','A12','A13','A15','A17','A18','A25','A27','B2','B3','B6','B8','B13']
	
	not_num_columns = ['A20','A28','B4','B9','B10','B11']
	for i in not_num_columns:
		data[i] = data[i].fillna('0:00-0:00')
		data[i] = data[i].str.replace('；|;|"|:{2}',':')
		data['nnum_'+i] = data[i].apply(get_time)
		num_columns.append('nnum_'+i)

	# 内部特征构造
	data[['A3','B1','B12']] = data[['A3','B1','B12']].fillna(0)
	data['in_sampleid'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))
	data['in_sum'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
	num_columns.append('in_sampleid')
	num_columns.append('in_sum')
	
	# 外部特征构造
	out=get_outFeature(data[['B14']])
	out_feature = [i for i in out.columns]
	data[out_feature]=out

	out_columns = out.columns.tolist()
	num_columns=num_columns+out_columns
	
	# 数据清洗&离散连续切分
	# checkNAN(data[num_columns])
	data=data.fillna(0)

	discrete_feature = [i for i in num_columns if i not in ['in_sampleid','in_sum']+out_columns]
	continuous_feature = [i for i in num_columns if i not in discrete_feature]

	# 离散型特征转换
	le = preprocessing.LabelEncoder()
	for i in discrete_feature:
		data[i] = le.fit_transform(data[i])

	ohe = preprocessing.OneHotEncoder()
	OneHot_data = ohe.fit_transform(data[discrete_feature])
	OneHot_data = pd.DataFrame(data=OneHot_data.toarray())
	discrete_feature = [i for i in OneHot_data.columns]
	data[discrete_feature]=OneHot_data

	# 连续型特征转换
	# 待填充

	# 特征选取
	if flag == 'train':
		data=data[discrete_feature+continuous_feature+['收率']]
	else:
		data=data[discrete_feature+continuous_feature+['样本id']]
	
	return data

if __name__ == "__main__":
	# 数据读取
	df_train = pd.read_csv('../data/testJinnan/In/jinnan_round1_train_20181227.csv',encoding='gb18030')
	df_test = pd.read_csv('../data/testJinnan/In/jinnan_round1_testA_20181227.csv',encoding='gb18030')
	
	# 训练测试数据axis=0时的异常值处理
	df_train = df_train[df_train['收率']>0.87]
	df_train = df_train[~df_train['A25'].str.contains('1900')]
	df_train = df_train[df_train['A22']!=3.5]
	df_train = df_train[df_train['B1']!=3.5]
	df_train['A20'] = df_train['A20'].str.replace('0分','0')
	df_train['A20'] = df_train['A20'].str.replace(':-','-')
	df_train['B4'] = df_train['B4'].str.replace(':-',':00-')
	df_train['B4'] = df_train['B4'].str.replace('1600','16:00')
	
	sourse=df_train.copy()

	# ******生成训练测试数据******
	df_train=get_feature(df_train.reset_index(drop=True),flag='train')
	df_test=get_feature(df_test.reset_index(drop=True),flag='test')
	print(df_train.head())
	print(df_test.head())

	# 输出临时数据集	
	df_train.to_csv('../data/testJinnan/temp/'+'data_train'+'.'+'csv',header=True,index=False)
	df_test.to_csv('../data/testJinnan/temp/'+'data_test'+'.'+'csv',header=True,index=False)