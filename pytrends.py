# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:07:08 2020

@author: sujay
"""

import pandas as pd
from pytrends.request import TrendReq

pytrend = TrendReq(hl='en-US', tz=360)
keywords = ['CIBC', 'FINANCE']
pytrend.build_payload(
     kw_list=keywords,
     cat=0,
     timeframe='today 3-m',
     geo='CA',
     gprop='')

#Interest Over time
data = pytrend.interest_over_time()

#Related Queries
rq = pytrend.related_queries()
rq.values()

#Related Topics
related_topic = pytrend.related_topics()
related_topic.values()

#Interest by Region
regiondata = pytrend.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)

#Save results to csv
regiondata.to_csv('Regiondata.csv',encoding='utf_8_sig')
data.to_csv('Py_VS_R.csv', encoding='utf_8_sig')