#-*- coding: utf-8 -*-

import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics

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