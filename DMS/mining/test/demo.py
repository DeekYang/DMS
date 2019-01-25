#-*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/Administrator.IC059UC5S1MOO1S/Documents/GitHub/DMS/DMS')
from mining import *
import pandas as pd
from datetime import datetime
import multiprocessing,time,os
import time

def task(pid):
    # do something
    print("Run task -%s" %(os.getpid()))
    start = time.time()
    result=pid
    end = time.time()
    print('Task runs %0.2f seconds.' %(end - start))

    return result

if __name__ == "__main__":
    source_offline = readDatafile('data/testO2O/In/ccf_offline_stage1_train.csv')
    source_offline['Date_received'],source_offline['Date'] = pd.to_datetime(source_offline['Date_received']
        ,format = '%Y%m%d'),pd.to_datetime(source_offline['Date'],format = '%Y%m%d')
    print(source_offline.info())
    source_offline = source_offline[((source_offline['Date']>'20160315') & (source_offline['Date']<'20160616')) 
    | ((pd.isnull(source_offline['Date'])) & ((source_offline['Date_received']>'20160315') & (source_offline['Date_received'] < '20160616')))]
    print(source_offline.info())

    # source_online = readDatafile('data/testO2O/In/ccf_online_stage1_train.csv')
    # print(source_online.info())
    # demo = readDatafile('data/testO2O/In/ccf_offline_stage1_test_revised.csv')
    # demo = demo.sort_values('Date_received',ascending=False)
    # print(demo.head())
    # print(demo.info())
    source_offline.to_csv('data/testO2O/In/'+'ccf_offline_stage1_train_temp'+'.'+'csv',header=True,index=False)
# source_online.to_csv('data/testO2O/In/'+'ccf_online_stage1_train_temp'+'.'+'csv',header=True,index=False)
# demo.to_csv('data/testO2O/In/'+'ccf_offline_stage1_test_revised_temp'+'.'+'csv',header=True,index=False)

# multiprocessing.freeze_support()
# pool = multiprocessing.Pool()
# cpus = multiprocessing.cpu_count()
# results = []
# for i in range(0, cpus):
#     result = pool.apply_async(task, args=(i,))
#     results.append(result)
# pool.close()
# pool.join()
# # print(results)
# for result in results:
    # print(result.get())


# df=pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
# df2=pd.DataFrame({'a':[7,7],'b':[7,7]})
# print(df)
# print(df2)
# data = pd.merge(df, df2,  how = 'left',left_index=True,right_index=True)
# print(data)

# t=df[df['a']>1].reset_index(drop=True,inplace=False)
# print(t)
# print(df['a'].unique())