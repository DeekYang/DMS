#-*- coding: utf-8 -*-

import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn import metrics
from sklearn.externals import joblib

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier

# 提取数据
data=readDatafile('temp/data.csv')
data=data.fillna(-1)

# print(data['label'].value_counts())

feature=[ 'Dis_type','Dis_rate','Dis_man','Dis_jian','Distance',
'u_nocoupon_count','u_withcoupon_out15day_count','u_withcoupon_in15day_count',
'm_nocoupon_count','m_withcoupon_out15day_count','m_withcoupon_in15day_count','weekday','day_type']

X=data[feature].as_matrix()
Y=data['label'].as_matrix()

skf = StratifiedKFold(n_splits=2,shuffle=True)
for train_index, test_index in skf.split(X,Y):

	x_train, x_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]

	# 建模
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
	# clf = AdaBoostClassifier(n_estimators=100)
	# clf = ensemble.RandomForestClassifier()
	model = Pipeline(steps=[('ss', StandardScaler()),('en', clf) ])
	model.fit(x_train, y_train)

	# 评估
	y_predict = model.predict(data[feature].as_matrix())  
	y_predict_p = model.predict_proba(data[feature].as_matrix()) 
	data['pred'] = y_predict
	data['pred_prob'] = y_predict_p[:, 1]

	# avgAUC calculation
	vg = data.groupby(['Coupon_id'])
	aucs = []
	for i in vg:
		tmpdf = i[1] 
		if len(tmpdf['label'].unique()) != 2:
			continue
		fpr, tpr, thresholds = metrics.roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
		aucs.append(metrics.auc(fpr, tpr))
	print(np.average(aucs))

	# print ('AUC:',metrics.auc(y_test,y_predict,reorder=True))
	# print ('roc_auc_score:',metrics.roc_auc_score(y_test, y_predict_p[:,0]))
	# print ('roc_curve:',metrics.roc_curve(y_test, y_predict_p[:,0]))

# 持久化
joblib.dump(model, 'temp/'+'filename.pkl') 