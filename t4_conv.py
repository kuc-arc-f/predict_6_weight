# -*- coding: utf-8 -*-
# csv変換、　＋Ｎ列
#

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from simple_net import SimpleNet
from util_dt import *
from util_df import *
import time

#
# 学習データ
wdata = pd.read_csv("dat_weight_2.csv" 
        ,names=("index" ,"weight", "height","mid_lenght","top_lenth") )

#print( wdata.head() )
wdata =add_arr_data(wdata )
wdata = add_index(wdata )
print(wdata.shape )
print(wdata.head() )

wdata.to_csv('dat_weight_4.csv', index=False)
quit()
