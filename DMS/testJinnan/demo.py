#-*- coding: utf-8 -*-
import sys
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
from sklearn.preprocessing import OneHotEncoder
	# 清洗：
	# 行数据阀值过滤

	# 特征：
	# 自动构建特征工具

	# 转换：
	# 分箱
if __name__ == "__main__":
	df_train = pd.read_csv('../data/testJinnan/In/jinnan_round1_train_20181227.csv',encoding='gb18030')
	# df_test = pd.read_csv('../data/testJinnan/In/jinnan_round1_testA_20181227.csv',encoding='gb18030')
	categorical_columns = [f for f in df_train.columns if f  in ['A1','A2']]
	# print(categorical_columns)
	# numerical_columns = [f for f in df_train.columns if f not in categorical_columns]
	# print(numerical_columns)

	# for f in categorical_columns:
		# print(dict(zip(df_train[f].unique(), range(0, df_train[f].nunique()))))
		# df_train[f] = df_train[f].map(dict(zip(df_train[f].unique(), range(0, df_train[f].nunique()))),na_action='ignore')
		# print(df_train[f].head())

	df_train = df_train[df_train['收率']>0.87]
	df_train['intTarget'] = pd.cut(df_train['收率'], 5, labels=False)
	# print(df_train)

	print(df_train['intTarget'].value_counts())
	print(df_train['intTarget'].value_counts(normalize=False).values)
	print(df_train['intTarget'].value_counts().values[0])

	# df_train = pd.get_dummies(df_train, columns=['intTarget'])
	# print(df_train.head())

	# li = ['intTarget_0','intTarget_2','intTarget_3','intTarget_4']
	# mean_columns = []
	# for f1 in categorical_columns:
	# 	for f2 in li:
	# 		col_name = 'B14_to_'+f1+"_"+f2+'_mean'
	# 		mean_columns.append(col_name)
	# 		order_label = df_train.groupby([f1])[f2].mean()
	# 		# print(order_label)
	# 		df_train[col_name] = df_train['B14'].map(order_label)
	# print(df_train.info())		


