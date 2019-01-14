#-*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

# 检查异常值情况--箱体图
def checkOutlier(data):
    print ("\n检查异常值情况:")   
    outlierdata=data.values#取数值  
    p = plt.boxplot(outlierdata)#箱体图
    # 显示异常数据的数值
    for i in range(outlierdata.shape[1]): 
        x=p['fliers'][i].get_xdata()
        y=p['fliers'][i].get_ydata()
        if len(x) != 0:
            y.sort()
            plt.annotate(y[0], xy = (x[0],y[0]), xytext = (x[0],y[0]))#显示最小值坐标
            plt.annotate(y[len(y)-1], xy = (x[len(y)-1],y[len(y)-1]), 
            xytext = (x[len(y)-1],y[len(y)-1]))#显示最大值坐标
    plt.show()

# 检查缺失值情况
def checkNAN(data):
    print ("\n检查缺失值情况:")
    nandata = pd.DataFrame(index = ['All','noNAN','NAN'],columns = data.count().index)
    nandata.loc[['All'],:] = data.shape[0]
    nandata.loc[['noNAN'],:] = data.count().values#data每列非NAN值的数量
    nandata.loc[['NAN'],:] = nandata.loc['All'].values-nandata.loc['noNAN'].values
    nandata = nandata.T[nandata.T['NAN'] > 0]
    if nandata.empty:
        print ("恭喜您,该数据集不存在缺失值！")
    else:
        print ("该数据集存在缺失值情况如下：")
        print (nandata)

# 检查重复值情况
def checkDuplicate(data):
    print ("\n检查重复值情况:")
    if data.shape[0] == data.drop_duplicates().shape[0]:
        print ("恭喜您,该数据集不存在重复值！")
    else:
        print ("该数据集存在 %d 行重复值！" % (data.shape[0]-data.drop_duplicates().shape[0]))

# 统计描述分析--直方图
def statistical(data):
    # 1.集中分析，均值 中位数 众数
    # 2.离中分析，极差 标准差方差 变异系数 四分位数间距
    statistics = data.describe() #保存基本统计量
    statistics.loc['var'] = statistics.loc['std']*statistics.loc['std'] #方差
    statistics.loc['range'] = statistics.loc['max']-statistics.loc['min'] #极差
    statistics.loc['cha'] = statistics.loc['std']/statistics.loc['mean'] #变异系数
    statistics.loc['dis'] = statistics.loc['75%']-statistics.loc['25%'] #四分位数间距
    print ("\n输出各变量的 均值 标准差 中位数 方差 极差 变异系数 四分位数间距：")
    print (statistics)
    print ("\n输出各变量的 众数：")
    print (data.mode())
    # 3.频率分布直方图
    # for i in data.columns:
    #     plt.hist(data[i],10)
    # plt.show()
    # 4.偏态%%峰态分析，正态分布图 标准正态分布图

# 变量关系分析--散点图，帕累托图
def correlation(data):
    # 1.散点图&&相关性分析
    for i in data.columns:
        plt.scatter(data.index,data[i],marker = '*') 
    plt.show()
    print ("\n各变量的相关系数为：")
    print (data.corr())#显示相关系数矩阵
    # print (data.corr()['num'])#只显示某个变量相对其它变量的相关系数
    # print (data['num'].corr(data['num2']))#显示某变量相对另一个变量的相关系数
    # # 2.帕累托图
    # data_pareto=data['num'].copy() #选取某一列数据作图
    # data_pareto=data_pareto.sort_values(ascending=False) #排序
    # data_pareto.plot(kind='bar')
    # # 累积值计算
    # p=1.0*data_pareto.cumsum()/data_pareto.sum()
    # p.plot(color = 'g', secondary_y = True, style = '-*',linewidth = 2,use_index=False)
    # # 标注临界点
    # for i in range(len(p)):
    #     if p[i]>0.8:
    #         break
    # print ("\n在第 %d 项 %s 处累积值已大于80%%临界值！" % (i+1,p.index[i]))
    # plt.annotate(format(p[i], '.2%'), xy = (12, p[i]), xytext=(12, p[i]*0.9), 
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")) 
    # plt.show()

# 对比分析--折线图
def contrast(data):
    # 1.绝对值分析
    # 2.相对值分析
    # 3.周期性分析
    for i in data.columns:   
        x=data.index
        y=data[i]
        plt.plot(x,y,label=i)
    plt.legend()   
    plt.show()

if __name__ == '__main__':
    data = pd.read_excel('../data/in_test.xls').head(20)
    # print(data)

    # checkOutlier(data)
    # checkNAN(data)
    # checkDuplicate(data)

    statistical(data)
    # correlation(data)
    # contrast(data)