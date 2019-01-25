#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *

if __name__ == "__main__":
	df_train = pd.read_csv('../data/testJinnan/In/jinnan_round1_train_20181227.csv',encoding='gb18030')
	df_test = pd.read_csv('../data/testJinnan/In/jinnan_round1_testA_20181227.csv',encoding='gb18030')
	df_test_B = pd.read_csv('../data/testJinnan/In/jinnan_round1_testB_20190121.csv',encoding='gb18030')

	# print(df_test_B.info())
	# print(df_test_B.head())

	# 缺失值
	# checkNAN(df)
	# checkNAN(df_test)

	# 重复值
	# df_Duplicate=df.iloc[:,1:-1].copy()
	# checkDuplicate(df_Duplicate)
	
	# 特征取值范围
	# statistical(df)
	# df_columns=df_train.iloc[:,1:-1].copy()
	# df_columns=df_columns[['A1','A2','A3','A4','A19','A21','A22','A23','B1','B12','B14']]
	# # df_columns=df_columns[['A8','A27','A6','A10','A12','A13','A15','A17','A18','A25','B2','B3','B6','B8','B13']]
	# df_columns=df_columns[['A19','B1','B12','B14']]
	# for i in df_columns.columns.tolist():
	# 	print(df_columns[i].unique())
	# 	print(df_columns[i].value_counts())

	# df_columns=df_test.iloc[:,1:].copy()
	# # df_columns=df_columns[['A1','A2','A3','A4','A19','A21','A22','A23','B1','B12','B14']]
	# df_columns=df_columns[['A8','A27','A6','A10','A12','A13','A15','A17','A18','A25','B2','B3','B6','B8','B13']]
	# df_columns=df_columns[['A25']]
	# for i in df_columns.columns.tolist():
	# 	print(df_columns[i].unique())
	# 	print(df_columns[i].value_counts())

	df_columns=df_test_B.iloc[:,1:].copy()
	# df_columns=df_columns[['A19','B1','B12','B14']]
	df_columns=df_columns[['A6','A10','A12','A15','A17','A22','A25','A27','B6','B8']]
	# df_columns=df_columns[['A25']]
	for i in df_columns.columns.tolist():
		print(df_columns[i].unique())
		print(df_columns[i].value_counts())

