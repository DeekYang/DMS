#-*- coding: utf-8 -*-
from urllib import request, parse
import pandas as pd
import json
import math
import time

# **********下载器**********
def get_page(url,headers):
	req = request.Request(url,headers=headers)		
	response = request.urlopen(req)	
	if response.getcode() == 200:
		# print (response.geturl()+'\n')#返回请求的url
		page = response.read().decode('utf-8')
		return page
	else:
		print ('Warning!')


# **********DEMO**********
if __name__ == "__main__": 
	# A.设置查询条件
	keywords = parse.quote('数据挖掘')
	city = parse.quote('杭州')
	district = parse.quote('')	
	
	# B.设置url&&header
	urlTest = r'http://www.lagou.com/jobs/positionAjax.json?kd='+keywords+'&city='+city+'&district='+district
	headers = {
		'Referer':'https://www.lagou.com/jobs/list_'+keywords,
		'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36',
		# 'Accept':'application/json, text/javascript, */*; q=0.01',
		# 'Accept-Encoding':'gzip, deflate, br',
		# 'Accept-Language':'zh-CN,zh;q=0.8,en;q=0.6,zh-TW;q=0.4',
		# 'Connection':'keep-alive',
		# 'Content-Length':'55',
		# 'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
		# 'Cookie':'_ga=GA1.2.921411534.1521191290; user_trace_token=20180316170813-8e3f7e88-28f9-11e8-b41f-525400f775ce; LGUID=20180316170813-8e3f8187-28f9-11e8-b41f-525400f775ce; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%22166a4c37f0d68-0346539b740345-333b5602-1049088-166a4c37f0f186%22%2C%22%24device_id%22%3A%22166a4c37f0d68-0346539b740345-333b5602-1049088-166a4c37f0f186%22%7D; _gid=GA1.2.245938301.1544192994; index_location_city=%E6%9D%AD%E5%B7%9E; LG_LOGIN_USER_ID=8fa657b360ac0da7fe8dda1fa982008c3ab5c2f0d5b81a27; showExpriedIndex=1; showExpriedCompanyHome=1; showExpriedMyPublish=1; hasDeliver=12; JSESSIONID=ABAAABAAADEAAFI09CEE6DDD7633F0EF81460C6911B5CC9; Hm_lvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1544192994,1544252148; _putrc=8888D3F260D1B76B; login=true; unick=%E6%9D%A8%E9%99%86%E5%B3%B0; gate_login_token=ea6132a3987e0b02f02a7179cc56674eb1836d2102f34c8a; TG-TRACK-CODE=index_navigation; LGSID=20181209000022-5e7ea5f8-fb02-11e8-8e4f-525400f775ce; PRE_UTM=; PRE_HOST=; PRE_SITE=; PRE_LAND=https%3A%2F%2Fwww.lagou.com%2Fjobs%2Flist_%25E6%2595%25B0%25E6%258D%25AE%25E6%258C%2596%25E6%258E%2598%3Fpx%3Ddefault%26city%3D%25E6%259D%25AD%25E5%25B7%259E; _gat=1; LGRID=20181209000829-8075ccb6-fb03-11e8-8e50-525400f775ce; Hm_lpvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1544285152; SEARCH_ID=7cfa7635d5324c5380338e43a6e44e31',
		# 'Host':'www.lagou.com',
		# 'Origin':'https://www.lagou.com',
		# 'X-Anit-Forge-Code':'0',
		# 'X-Anit-Forge-Token':'None',
		# 'X-Requested-With':'XMLHttpRequest'
	}

	# C.获取查询结果总数
	data=get_page(urlTest,headers)
	totalCount = int(json.loads(str(data))["content"]["positionResult"]["totalCount"])
	print('***本次搜索到%d个职位***'%totalCount)
	pagenum = int(math.ceil(totalCount/15))

	# D.分页查询各页数据值&&不同数据格式的转存
	pagenum=0 #启用flag
	for i in range(0,pagenum):
		url=urlTest+'&pn='+str(i+1)
		data=get_page(url,headers)	
		jsondata = json.loads(str(data))['content']['positionResult']['result']	
		for t in list(range(len(jsondata))):
			jsondata[t].pop('companyLogo','city')
			jsondata[t].pop('explain')
			jsondata[t].pop('plus')
			jsondata[t].pop('gradeDescription')
			jsondata[t].pop('promotionScoreExplain')
			jsondata[t].pop('adWord')
			jsondata[t].pop('appShow')
			jsondata[t].pop('approve')
			jsondata[t].pop('companyId')
			jsondata[t].pop('deliver')
			jsondata[t].pop('pcShow')
			jsondata[t].pop('positionId')
			jsondata[t].pop('score')
			jsondata[t].pop('publisherId')
			if t == 0:
				rdata=pd.DataFrame(pd.Series(data=jsondata[t])).T
			else:
				rdata=pd.concat([rdata,pd.DataFrame(pd.Series(data=jsondata[t])).T])
		if i == 0:
			citydata=rdata
		else:
			citydata=pd.concat([citydata,rdata])
		time.sleep(10)	
	
		# 保存查询结果
		citydata.to_excel('LagouSpider.xls',sheet_name='sheet1')