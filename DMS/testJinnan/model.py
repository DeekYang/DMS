#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
import lightgbm as lgb

# *****提取数据&拆分&抽样*****
data = readDatafile('../data/testJinnan/temp/data_train.csv')
# print(data.info())
X,Y=data.iloc[:,:-1].copy(),data['收率'].copy()

cv = KFold(n_splits=3,shuffle=True)

# *****模型调参*****
# print(GradientBoostingRegressor().get_params())
tuned_parameters = {
					'learning_rate':[0.1,0.8],
					'n_estimators':[200],
					}
clf = GradientBoostingRegressor(n_estimators=200)
model = GridSearchCV(clf, tuned_parameters, cv=cv, scoring='neg_mean_squared_error')
model.fit(X,Y)
print('Best_params:',model.best_params_)
print(model.best_score_)
print(type(model.cv_results_))

# 模型融合
# print('模型融合')


# *****交叉验证建模评估*****
# clf = GradientBoostingRegressor(n_estimators=200)
# clf = lgb.LGBMRegressor()
# out = cross_val_score(clf, X, Y, cv=cv , scoring='neg_mean_squared_error')
# print(out.mean())

'''
# *****手工建模评估*****
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
	x_test['pred'] = y_predict
	out.append(metrics.mean_squared_error(y_test, x_test['pred']))
print(np.array(out).mean())
'''

# *****模型持久化*****
joblib.dump(model, '../data/testJinnan/temp/'+'filename.pkl')
