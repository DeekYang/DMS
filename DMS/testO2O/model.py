#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn import metrics
from sklearn.externals import joblib

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier

# 提取数据
data=readDatafile('../data/testO2O/temp/data_train.csv')

# 选取特征
feature=['Coupon_id','Weekday','Day_type',
'Dis_type','Dis_rate','Dis_man','Dis_jian',
'u_coupon_count','u_buy_count','u_buy_with_coupon','u_merchant_count',
'u_use_coupon_rate','u_buy_with_coupon_rate',
'Distance','u_min_distance','u_max_distance','u_mean_distance','u_median_distance',
'label']
data=data[feature]
# print(data.head())

# 抽样
skf = StratifiedKFold(n_splits=5,shuffle=True)
# 拆分
X,Y=data.iloc[:,0:-1].copy(),data['label'].copy()
for train_index, test_index in skf.split(X,Y):
	x_coupon_id = X.iloc[test_index,:1]
	x_train, x_test = X.iloc[train_index,1:], X.iloc[test_index,1:]
	y_train, y_test = Y[train_index], Y[test_index]

	# 训练
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
	# clf = AdaBoostClassifier(n_estimators=100)
	model = Pipeline(steps=[('en', clf)])
	model.fit(x_train, y_train)

	# 预测
	y_predict_p = model.predict_proba(x_test)
	
	# 评估 
	x_test = x_test.join(x_coupon_id)
	x_test = x_test.join(y_test)
	x_test['pred_prob']=y_predict_p[:,1]
	
	vg = x_test.groupby(['Coupon_id'])
	aucs = []
	for i in vg:		
		tmpdf = i[1] 		
		if len(tmpdf['label'].unique()) != 2:
			continue
		aucs.append(metrics.roc_auc_score(tmpdf['label'], tmpdf['pred_prob']))
	print(np.average(aucs))

# 模型持久化
# joblib.dump(model, '../data/testO2O/temp/'+'filename.pkl') 