#-*- coding: utf-8 -*-

import pandas as pd
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


# if __name__ == '__main__':
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
    
    