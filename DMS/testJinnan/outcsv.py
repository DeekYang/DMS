#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
from sklearn.externals import joblib

if __name__ == "__main__":
	# 提取数据&&模型
	dftest = readDatafile('../data/testJinnan/temp/data_test.csv')
	clf = joblib.load('../data/testJinnan/temp/filename.pkl')
	print(dftest.head())
	# 预测
	dftest['target'] = clf.predict(dftest.iloc[:,:-1]).round(3)
	
	# 运用输出
	dftest=dftest[['样本id','target']].copy()
	dftest.to_csv('../data/testJinnan/Out/'+'submit.csv', index=False, header=False)