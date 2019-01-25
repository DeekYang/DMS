#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
from sklearn.externals import joblib
from sklearn import metrics

if __name__ == "__main__":

	dftest = pd.read_csv('../data/testJinnan/Out/submit.csv')
	dftest2 = pd.read_csv('../data/testJinnan/Out/submit20190122.csv')

	# print(dftest.head())
	# print(dftest2.head())

	df=pd.merge(dftest,dftest2, how = 'left',on='样本id')
	# print(df.shape)
	print(df.head())
	print('score: {:<8.8f}'.format(metrics.mean_squared_error(df['target_x'], df['target_y'])))
