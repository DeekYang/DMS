#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *

if __name__ == "__main__":
	df = pd.read_csv('../data/testJinnan/In/jinnan_round1_train_20181227.csv',encoding='gb18030')
	df_test = pd.read_csv('../data/testJinnan/In/jinnan_round1_testA_20181227.csv',encoding='gb18030')
	# print(df.info())
	# print(df.head())

	# 缺失值
	# checkNAN(df)
	# checkNAN(df_test)

	# 重复值
	# df_Duplicate=df.iloc[:,1:-1].copy()
	# checkDuplicate(df_Duplicate)
	
	# 特征取值范围
	# statistical(df)
	# df_columns=df.iloc[:,1:-1].copy()
	# df_columns=df_columns[['A1','A2','A3','A4','A19','A21','A22','A23','B1','B12','B14']]
	# # df_columns=df_columns[['A8']]
	# for i in df_columns.columns.tolist():
	# 	print(df_columns[i].unique())
	# 	print(df_columns[i].value_counts())

	D=df.iloc[:,1:6].copy()
	contrast(D)

