#-*- coding: utf-8 -*-
from mining import *


if __name__ == "__main__":
	df_train = pd.read_csv('../../data/project/testJinnan/In/jinnan_round1_train_20181227.csv',encoding='gb18030')
	df_test_A = pd.read_csv('../../data/project/testJinnan/In/jinnan_round1_testA_20181227.csv',encoding='gb18030')
	df_test_B = pd.read_csv('../../data/project/testJinnan/In/jinnan_round1_testB_20190121.csv',encoding='gb18030')

	# print(df_train.info())
	# print(df_train.head())

	# 缺失值
	# checkNAN(df_train)

	# 重复值
	# checkDuplicate(df_train)
	
	# 特征取值范围
	# statistical(df_train)

	df_columns=df_train.iloc[:,1:].copy()
	# df_columns=df_columns[['A19','B1','B12','B14']]
	df_columns=df_columns[['A6','A10','A12','A15','A17','A22','A25','A27','B6','B8']]
	for i in df_columns.columns.tolist():
		print(df_columns[i].unique())
		print(df_columns[i].value_counts())

	# df_columns=df_test_A.iloc[:,1:].copy()
	# df_columns=df_columns[['A1','A2','A3','A4','A19','A21','A22','A23','B1','B12','B14']]
	# df_columns=df_columns[['A8','A27','A6','A10','A12','A13','A15','A17','A18','A25','B2','B3','B6','B8','B13']]
	# for i in df_columns.columns.tolist():
	# 	print(df_columns[i].unique())
	# 	print(df_columns[i].value_counts())