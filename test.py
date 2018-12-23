# -*- coding: utf-8 -*-
# DL, pred_2-ML の, DL 版(体重の予測)
# train/学習処理。結果ファイル保存。
# TwoLayerNet を参考に、３層ネットワーク利用
#  学習　>パラメータ保存

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from simple_net import SimpleNet
from util_dt import *
import time

#
#arr = np.random.rand(100 ) * 40 +30
#print(arr ) 
#
a1 = {'ID':['11','12','13' ]
        ,'city':['Tokyo','Osaka','Kyoto' ]
        ,'num1':[ 101 ,102,103 ]
        }

df = DataFrame(a1 )
num_df =df.shape[0]

List =[]
for i in range(num_df):
    ct= i+ 1
#    print(i )
    List.append(ct )
#print(List)

df["idx"] =List
print(df.shape )
print(df.shape[0] )

print(df.head() )
quit()
#print(df )
for item in df:
    print(item)

