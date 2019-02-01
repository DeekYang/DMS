#-*- coding: utf-8 -*-

import pandas as pd
from mining import *

# 读取数据
data_offline=readDatafile('data/ccf_offline_stage1_train.csv').head(10)
data_offline=data_offline[data_offline['Coupon_id'].notnull()]

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

# 数据建模与评估

# 模型融合