#-*- coding: utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib


if __name__ == "__main__":
	# 提取模型&&数据
	model = joblib.load('../../data/project/testJinnan/temp/filename.pkl')
	dftest_A = pd.read_csv('../../data/project/testJinnan/temp/data_test_A.csv')		
	dftest_B = pd.read_csv('../../data/project/testJinnan/temp/data_test_B.csv')
	
	# 预测
	dftest_A['target'] = model.predict(dftest_A.iloc[:,:-1]).round(3)
	dftest_B['target'] = model.predict(dftest_B.iloc[:,:-1]).round(3)
	
	# 对比实际数值
	dftrue_A = pd.read_csv('../../data/project/testJinnan/Out/jinnan_round1_ansA_20190125.csv',names=['样本id','target'])
	df_A=pd.merge(dftest_A,dftrue_A, how = 'left',on='样本id')
	print('A score: {:<8.8f}'.format(metrics.mean_squared_error(df_A['target_x'], df_A['target_y'])*0.5))
	dftrue_B = pd.read_csv('../../data/project/testJinnan/Out/jinnan_round1_ansB_20190125.csv',names=['样本id','target'])
	df_B=pd.merge(dftest_B,dftrue_B, how = 'left',on='样本id')
	print('B score: {:<8.8f}'.format(metrics.mean_squared_error(df_B['target_x'], df_B['target_y'])*0.5))

	# 运用输出
	# dftest=dftest[['样本id','target']].copy()
	# dftest.to_csv('../../data/project/testJinnan/Out/'+'submit.csv', index=False, header=False)

	