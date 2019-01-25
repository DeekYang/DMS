#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier,AdaBoostRegressor,RandomForestClassifier
import lightgbm as lgb


# *****交叉验证-手工建模评估*****
def cross_val_artificial(X,Y,cv):
	out = []
	for train_index, test_index in cv.split(X):
		x_train,x_test = X.iloc[train_index].copy(),X.iloc[test_index].copy()
		y_train,y_test = Y.iloc[train_index].copy(),Y.iloc[test_index].copy()
		# 训练
		clf = GradientBoostingRegressor(n_estimators=200)
		clf.fit(x_train, y_train)
		# 预测
		y_predict = clf.predict(x_test)
		# 评估 
		x_test['pred'] = y_predict.round(3)
		n=x_test.join(y_test)
		n['n']=n['pred']-n['收率']
		print(n[['in_sampleid','收率','pred','n']].sort_values(by='n'))
		
		out.append(metrics.mean_squared_error(y_test, x_test['pred']))
	print("CV score: {:<8.8f}".format(np.array(out).mean()*0.5))

# *****交叉验证-cross_val_score建模评估*****
def cross_val_function(X,Y,cv):
	clf = GradientBoostingRegressor(n_estimators=200)
	# clf = lgb.LGBMRegressor()
	out = cross_val_score(clf, X, Y, cv=cv , scoring='neg_mean_squared_error')
	print("CV score: {:<8.8f}".format(out.mean()*0.5))

# *****交叉验证-lightGBM*****
def cross_val_lightGBM(X,Y,cv):
	param = {'num_leaves': 120,
			 'min_data_in_leaf': 30, 
			 'objective':'regression',
			 'max_depth': -1,
			 'learning_rate': 0.01,
			 # "min_child_samples": 30,
			 "boosting": "gbdt",
			 "feature_fraction": 0.9,
			 "bagging_freq": 1,
			 "bagging_fraction": 0.9 ,
			 "bagging_seed": 11,
			 "metric": 'mse',
			 "lambda_l1": 0.1,
			 "verbosity": -1}

	oof_lgb = np.zeros(len(X))

	for fold_, (trn_idx, val_idx) in enumerate(cv.split(X, Y)):
		print("fold n°{}".format(fold_+1))
		trn_data = lgb.Dataset(X.iloc[trn_idx], Y.iloc[trn_idx])
		val_data = lgb.Dataset(X.iloc[val_idx], Y.iloc[val_idx])

		num_round = 10000
		clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
		oof_lgb[val_idx] = clf.predict(X.iloc[val_idx], num_iteration=clf.best_iteration)
	
	print("CV score: {:<8.8f}".format(metrics.mean_squared_error(oof_lgb, Y)*0.5))

# *****模型调参-获取模型超参数*****
def get_hyper_parameters(X,Y,cv):
	# print(GradientBoostingRegressor().get_params())
	tuned_parameters = {
						# 'n_estimators':range(20,81,10),
						# 'learning_rate':[],
						# 'subsample':[],
						# 'init':[],
						# 'loss':[],
						# 'alpha':[],
						# 'max_features':[],
						# 'max_depth':range(3,14,2),
						# 'min_samples_split':range(20,80,20),
						'min_samples_leaf':range(10,50,10),
						# 'min_weight_fraction_leaf':[],
						# 'max_leaf_nodes':[],
						# 'min_impurity_split':[],
						}
	clf = GradientBoostingRegressor(n_estimators=200)
	model = GridSearchCV(clf, tuned_parameters, cv=cv, scoring='neg_mean_squared_error')
	model.fit(X,Y)
	# print(model.cv_results_)
	print('Best_params:',model.best_params_)
	print(model.best_score_)
	


if __name__ == "__main__":
	
	# 提取&拆分&抽样数据
	data = readDatafile('../data/testJinnan/temp/data_train.csv')
	# print(data.shape)
	# t=[i for i in data.columns if i not in ['in_sum_num','in_sum_time']]
	# data=data[t].copy()
	X,Y=data.iloc[:,:-1].copy(),data['收率'].copy()
	
	cv = KFold(n_splits=3,shuffle=True,random_state=2018)

	# 交叉验证
	# cross_val_artificial(X,Y,cv)
	# cross_val_function(X,Y,cv)
	# cross_val_lightGBM(X,Y,cv)

	# 模型调参
	# get_hyper_parameters(X,Y,cv)

	# 正式建模&模型持久化
	# model = GradientBoostingRegressor(n_estimators=80 ,learning_rate=0.1,max_depth=11,min_samples_split=60)
	model = GradientBoostingRegressor(n_estimators=200)
	model.fit(X,Y)
	# joblib.dump(model, '../data/testJinnan/temp/'+'filename.pkl')
	
	# 模型融合
	# print('模型融合！')

