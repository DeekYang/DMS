#-*- coding: utf-8 -*-
from mining import *
from sklearn import metrics
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
import lightgbm as lgb
import xgboost as xgb

if __name__ == "__main__":
	# 提取
	data = readDatafile('../data/testJinnan/temp/data_train.csv')
	
	# 拆分
	X,Y = data.iloc[:,:-1].copy(),data.iloc[:,-1].copy()	
	
	# 抽样-交叉验证
	CV = KFold(n_splits=5,shuffle=True,random_state=2019)	
	
	# 模型选取
	model_List = {  'GBRT': GradientBoostingRegressor(),
					'GBRT2': GradientBoostingRegressor(n_estimators=300,max_features=90),
					'XGBoost': xgb.XGBRegressor(),
					'XGBoost2': xgb.XGBRegressor(n_estimators=400,max_depth=4,colsample_bytree=0.5,n_jobs=2),
					'LightGBM': lgb.LGBMRegressor(),
					'LightGBM2': lgb.LGBMRegressor(n_estimators=80,n_jobs=2),
				 }
	Bestmodel = get_model_score(X,Y,CV,model_List)

	# 模型调参
	# bestmodel = get_hyper_parameters(X,Y,CV)

	# 模型持久化
	save_model(X,Y,Bestmodel)