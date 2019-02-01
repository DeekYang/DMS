#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sqlalchemy import create_engine

# 数据文件输入in
def readDatafile(filename):
    if ('txt' in filename):
        data = pd.read_table(filename)
    if ('csv' in filename):
        data = pd.read_csv(filename)
    if ('xlsx' in filename or 'xls' in filename):
        data = pd.read_excel(filename)
    return data

# 数据库输入in
def readDatabase(databaseURL,tablename):
    engine=create_engine(databaseURL)
    data=pd.read_sql(tablename,engine)#分布式读取？
    return data

# 数据文件输出out
def toDatafile(df,filename,filetype):
    if(filetype == 'txt'):
        df.to_csv('../data/out_'+filename+'.'+filetype,sep='\t',header=True,index=True)
    if(filetype == 'csv'):
        df.to_csv('../data/out_'+filename+'.'+filetype,header=True,index=True)      
    if(filetype =='xlsx' or filetype == 'xls'):
        df.to_excel('../data/out_'+filename+'.'+filetype,header=True,index=True)

# 数据库输出out
def toDatabase(df,databaseURL,tablename):
    engine=create_engine(databaseURL)
    df.to_sql(tablename,engine,if_exists='replace')

# 数据转换
def changeContinuity(data):
    waychange = input('请选择数据转换处理方式：1.标准化;2.归一化;3.二值化:')  
    # # demo:
    # X_train = np.array([[ 1., -1.,  2.],
    #                      [ 2.,  0.,  0.],
    #                      [ 0.,  1., -1.]])
    # print(pd.DataFrame(X_train).describe())#样本标准偏差,偏大

    # X_scaled = preprocessing.scale(X_train)
    # print (X_scaled)
    # print (X_scaled.mean(axis=0))
    # print (X_scaled.std(axis=0))#总体样本偏差

    # scaler = preprocessing.StandardScaler(with_mean=True,with_std=True).fit(X_train)
    # print (scaler)
    # print (scaler.mean_)
    # print (scaler.scale_)
    # print (scaler.transform(X_train))
    # X_test = [[-1., 1., 0.]]
    # print (scaler.transform(X_test))
    #数据缩放
    if waychange == '1':
        # 标准化
        scaled = preprocessing.scale(data)
        return scaled
    elif waychange == '2': 
        # 归一化
        normalized = preprocessing.normalize(data, norm='l2')
        return normalized
    else:
        # 二值化
        threshold=input('请输入阀值:')

        binarized = preprocessing.binarize(data,threshold=float(threshold))
        return binarized

def changeDisperse(data):
    # 一位有效编码（One-Hot-Encoding）
    # 标签编码
    return data

# 处理异常值
def handleOutlier(data):
    wayoutlier = input('请选择异常值的处理方式：1.替换为NAN值;2.不处理:')  
    print(wayoutlier)
    if wayoutlier == '1':
        # 1.替换为NAN值
        for i in data.columns:
            min=data[i].quantile(0.25)-(data[i].quantile(0.75)-data[i].quantile(0.25))*1.5
            max=data[i].quantile(0.75)+(data[i].quantile(0.75)-data[i].quantile(0.25))*1.5
            for j in range(len(data)):
                if (data[i][j]<min or data[i][j]>max):#通过四分位距方式判断是否为异常值
                    data[i]=data[i].replace(to_replace=data[i][j],value=np.nan)
    else:
        # 2.不处理
        print ("不处理异常值！\n")

# 处理缺失值
def handleNAN(data):
    waynan = input('请选择缺失值的处理方式：1.替换;2.删除;3.不处理:')  
    if waynan == '1':
        # 1.替换缺失值
        for i in data.columns: 
            avr=data[i].mean()#用算术平均数替换缺失值
            #2.0可扩展功能(1固定值2均值中位数众数3临近值4回归方法5拉格朗日插值法，牛顿插值法，分段差值，Hermite插值，样条插值法)
            data[i]=data[i].fillna(avr)
    elif waynan == '2':
        # 2.删除缺失值
        data=data.dropna(axis='index',how='any')
    else:
        # 3.不处理
        print ("不处理缺失值！\n")

# 处理重复值
def handleDuplicate(data):
    data=data.drop_duplicates()
    return data

# 属性规约
def selectColumns(data):
    pca=PCA()
    pca.fit(data)
    print ('各个特征向量：')
    print (pca.components_)
    print ('各个特征方差百分比（贡献率）：')
    print (pca.explained_variance_ratio_)
    # low_d = pca.transform(data)
    # print (low_d)
    # return low_d

# 数据规约
def selectRows(data):
    return data
    
if __name__ == '__main__':
    # databaseURL='mysql+pymysql://root:root@127.0.0.1:3306/database?charset=utf8'
    
    # 读取数据文件
    # df=readDatafile('../data/in_test.txt')
    # print (df.head())
    
    # 读取数据库数据
    # db=readDatabase(databaseURL,'test')
    # print (db.head())

    # 输出数据
    # toDatafile(df.head(),'filename','xlsx')
    # toDatabase(df.head(),databaseURL,'filename')


    data = pd.read_excel('../data/in_test.xls').head(10)
    print (data)
    data = changeContinuity(data)
    # data = changeDisperse(data)
    # data = handleOutlier(data)
    # data = handleNAN(data)
    # data = handleDuplicate(data)
    # data = selectColumns(data)
    # data = selectRows(data)
    print (data)