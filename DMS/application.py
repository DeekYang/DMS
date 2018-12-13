#-*- coding: utf-8 -*-

import pandas as pd
from mining import *

# 读取数据
data_offline=readDatafile('data/ccf_offline_stage1_train.csv').head(10)
data_offline=data_offline[data_offline['Coupon_id'].notnull()]
# data_online=readDatafile('data/ccf_online_stage1_train.csv')
print (data_offline.count())
# print (data_online.count())

data_offline.loc[data_offline['Date'].isnull(),'target'] = 0
data_offline.loc[data_offline['Date'].notnull(),'target'] = 1
# if data_offline['Date']=='NAN':
# 	data_offline['target']=1
print (data_offline)
# print (data_offline.describe())
# 质量分析
# checkOutlier(data)
# checkNAN(data)
# checkDuplicate(data)

# 特征分析
# statistical(data)
# contrast(data)
# correlation(data)

# 数据转换
# data = changeOfContinuity(data)
# data = changeOfDisperse(data)

# 数据清洗
# data = handleOutlier(data)
# data = handleNAN(data)
# data = handleDuplicate(data)

# 数据规约
# data = selectColumns(data)
# data = selectRows(data)		

# print (data)

# 数据建模与评估

# 模型融合