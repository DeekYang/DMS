#-*- coding: utf-8 -*-

import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics

# *****交叉验证-手工建模评估*****
def cross_val_artificial(x,y,cv,model):
    out = []
    for fold,(train_index, test_index) in enumerate(cv.split(X, Y),start=1):
        x_train,x_test = X.iloc[train_index].copy(),X.iloc[test_index].copy()
        y_train,y_test = Y.iloc[train_index].copy(),Y.iloc[test_index].copy()       
        # 训练
        model.fit(x_train, y_train)
        # 预测
        y_predict = model.predict(x_test).round(3)
        # 评估 
        score = metrics.mean_squared_error(y_test, y_predict)
        out.append(score)
        # print("fold n°{}: {:<8.8f}".format(fold,score))
    print("CV score: {:<8.8f}".format(np.array(out).mean()*0.5))

# *****模型评估选取*****
def get_model_score(x,y,cv,model_list):
    basescore = -1
    
    for model in model_list:
        print(model+' score: ')
        out = cross_val_score(model_list[model], x, y, cv=cv , scoring='neg_mean_squared_error')
        score = out.mean()*0.5
        print("{:<8.8f}".format(score)) 
        if score > basescore:
            basescore = score
            bestmodel = model_list[model]   
    return bestmodel

# *****模型调参-获取模型超参数*****
def get_hyper_parameters(x,y,cv):
    clf = xgb.XGBRegressor(n_estimators=400,max_depth=4,colsample_bytree=0.5,n_jobs=2)
    # print(clf.get_params())
    # parameters = {
                        # 'alpha':[],
                        # 'criterion':[],
                        # 'init':[],
                        # 'learning_rate':[0.01,0.05,0.1],
                        # 'loss':[],
                        # 'max_depth':[],
                        # 'max_features':range(80,125,10),
                        # 'max_leaf_nodes':[],
                        # 'min_impurity_decrease':[],
                        # 'min_impurity_split':[],
                        # 'min_samples_leaf':[],
                        # 'min_samples_split':[],
                        # 'min_weight_fraction_leaf':[],    
                        # 'n_estimators':range(200,601,100),
                        # 'n_iter_no_change': [],
                        # 'presort': [],
                        # 'random_state': [],
                        # 'subsample':[],
                        # 'tol': [],
                        # 'validation_fraction': [],
                        # 'verbose': [],
                        # 'warm_start': [],                     
                        # }
    parameters = {
                        # 'base_score':[],
                        # 'booster':['gbtree','gblinear','dart'],
                        # 'colsample_bylevel':[],
                        # 'colsample_bytree':[0.3,0.5,0.7],
                        # 'gamma':[],
                        # 'learning_rate':[0.05,0.1,0.2],
                        # 'max_delta_step':[],
                        # 'max_depth':range(3,6,1),
                        # 'min_child_weight':[],
                        # 'missing':[],
                        # 'n_estimators':range(200,601,100),
                        # 'n_jobs':[],
                        # 'nthread':[],
                        # 'objective':[],   
                        # 'random_state':[],
                        # 'reg_alpha': [],
                        # 'reg_lambda': [],
                        # 'scale_pos_weight': [],
                        # 'seed':[],
                        # 'silent': [],
                        # 'subsample': [],                  
                        }
    # parameters = {
                        # 'boosting_type':[],
                        # 'class_weight':[],
                        # 'colsample_bytree':[],
                        # 'importance_type':[],
                        # 'learning_rate':[0.05,0.1,0.2],   
                        # 'max_depth':range(5,10,1),
                        # 'min_child_samples':[],
                        # 'min_child_weight':[],
                        # 'min_split_gain':[],
                        # 'n_estimators':range(50,100,10),
                        # 'n_jobs':[],
                        # 'num_leaves':[],
                        # 'objective':[],   
                        # 'random_state':[],
                        # 'reg_alpha': [],
                        # 'reg_lambda': [],
                        # 'silent': [],
                        # 'subsample': [],
                        # 'subsample_for_bin': [],
                        # 'subsample_freq':[],                  
                        # }

    model = GridSearchCV(clf, parameters, cv=cv, scoring='neg_mean_squared_error',return_train_score=True)
    model.fit(X,Y)

    # print(model.cv_results_)
    print('params:',model.cv_results_['params'])

    print('mean_test_score:',model.cv_results_['mean_test_score']*0.5)
    print('std_test_score:',model.cv_results_['std_test_score']*0.5)
    print('mean_train_score:',model.cv_results_['mean_train_score']*0.5)
    print('std_train_score:',model.cv_results_['std_train_score']*0.5)

    print('best_params:',model.best_params_)
    print('best_score: {:<8.8f}'.format(model.best_score_*0.5))

# *****模型持久化*****
def save_model(x,y,model):
    model.fit(X,Y)
    joblib.dump(model,'../data/testJinnan/temp/'+'filename.pkl')

# 创建数据集
def loadData(filename,length):
    data = pd.read_table(filename).head(length)
    data = data.as_matrix()
    #训练数据
    train = data[:int(0.8*len(data))]
    x_train = train.T[:-1].T
    y_train = train.T[-1:].T.reshape((-1))
    #测试数据
    test = data[int(0.8*len(data)):]
    x_test = test.T[:-1].T
    y_test = test.T[-1:].T
    return x_train,y_train,x_test,y_test

# 监督学习-回归模型评估
def regressionModel(x_train,y_train,x_test,y_test):
    model_choice = pd.DataFrame([[1,'LinearRegression',1],[2,'SVM_SVR',1],
    [3,'DecisionTreeRegressor',1],[4,'GradientBoosting',1]],#设置是否启用模型，1启用：0停用
        columns = ['num','modelname','enable'])
    for x in range(len(model_choice)):   
        if model_choice.iloc[x:x+1,2:3].values == 1:
            model_num = model_choice.iloc[x:x+1,0:1].values
            if model_num == 1 : 
                model = linear_model.LinearRegression()
            elif model_num == 2 : 
                model = svm.SVR()
            elif model_num == 3 : 
                model = tree.DecisionTreeRegressor()
            else : 
                model = ensemble.GradientBoostingRegressor()
            model.fit(x_train,y_train)    
            y_predict = model.predict(x_test)      
            print ('MSE:',model_choice.iloc[x:x+1,1:2].values,
                metrics.mean_squared_error(y_test.flatten(),y_predict.flatten()))

# 监督学习-分类模型评估
def classificationModel(x_train,y_train,x_test,y_test):
    model_choice = pd.DataFrame([[1,'logisticRegression',1],[2,'SVM_SVC',1],
    [3,'DecisionTreeClassifier',1],[4,'RandomForestClassifier',1]],#设置是否启用模型，1启用：0停用
        columns = ['num','modelname','enable'])
    for x in range(len(model_choice)):   
        if model_choice.iloc[x:x+1,2:3].values == 1:
            model_num = model_choice.iloc[x:x+1,0:1].values
            if model_num == 1 : 
                model = linear_model.LogisticRegression()
            elif model_num == 2 : 
                model = svm.SVC()
            elif model_num == 3 : 
                model = tree.DecisionTreeClassifier()
            else : 
                model = ensemble.RandomForestClassifier()
            model.fit(x_train,y_train)    
            y_predict = model.predict(x_test)       
            print ('AUC:',model_choice.iloc[x:x+1,1:2].values,
                metrics.auc(y_test.flatten(),y_predict.flatten()))

# 非监督学习
# 建设中。。。


'''********主函数********'''

if __name__ == '__main__':
    print ('模型评估结果如下：')
    x_train,y_train,x_test,y_test=loadData('../data/in_test.txt',10)#读取预处理后数据集
    
    print ('回归模型：')
    regressionModel(x_train,y_train,x_test,y_test)
    
    print ('分类模型：')
    classificationModel(x_train,y_train,x_test,y_test)