# coding: utf-8
# util, dataframe

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import time
import datetime

#
def add_index(arr):
    num_df =arr.shape[0]

    List =[]
    for i in range(num_df):
        ct= i+ 1
        #    print(i )
        List.append(ct )
    #print(List)

    arr["index"] =List
    return arr
    #print(df.shape )
    #print(df.shape[0] )

#
def add_arr_data(arr):
    num_add= 1000
    height=arr["height"]
    max_height = height.max()
    min_height = height.min()
    height_add =np.random.rand(num_add ) * (max_height- min_height ) +min_height
    print(max_height )
    print(min_height )
#    print(height_add )
    mid_lenght=arr["mid_lenght"]
    max_mid_lenght = mid_lenght.max()
    min_mid_lenght = mid_lenght.min()
    mid_lenght_add =np.random.randint(min_mid_lenght, max_mid_lenght, num_add  ) 
    mid_lenght_add =np.random.rand(num_add ) * (max_mid_lenght- min_mid_lenght ) +min_mid_lenght
#    print(mid_lenght_add )

    top_lenth= arr["top_lenth"]
    max_top_lenth = top_lenth.max()
    min_top_lenth = top_lenth.min()
#    top_lenth_add =np.random.randint(min_top_lenth, max_top_lenth, num_add  )
    top_lenth_add =np.random.rand(num_add ) * (max_top_lenth- min_top_lenth ) +min_top_lenth
    #print(top_lenth_add) 
    #arr["height"]= height_add    
    #
    weight=arr["weight"]
    #weight= arr
    max_weight = weight.max()
    min_weight = weight.min()
    print(max_weight)
    print(min_weight)
    weight_add =np.random.randint(min_weight , max_weight, num_add  ) 
    #   
    a1 = {'height':height_add
        , 'mid_lenght':mid_lenght_add
        , 'top_lenth': top_lenth_add
        , 'weight': weight_add
        }
    df = DataFrame(a1 )
    #print(df.head() )
    return df 

#
# convert, a
def proc_add_arr(arr ):
    num_add= 1000
    height=arr["height"]
    max_height = height.max()
    min_height = height.min()
    #height_add =np.random.randint(min_height, max_height, num_add  ) 
    height_add =np.random.rand(num_add ) * (max_height- min_height ) +min_height
#    arr = np.random.rand(100 ) * 40 +30
#    max_height = arr["height"].max()
#    min_height = arr["height"].min()
    print(max_height )
    print(min_height )
#    print(height_add )
    mid_lenght=arr["mid_lenght"]
    max_mid_lenght = mid_lenght.max()
    min_mid_lenght = mid_lenght.min()
    mid_lenght_add =np.random.randint(min_mid_lenght, max_mid_lenght, num_add  ) 
    mid_lenght_add =np.random.rand(num_add ) * (max_mid_lenght- min_mid_lenght ) +min_mid_lenght
#    print(mid_lenght_add )

    top_lenth= arr["top_lenth"]
    max_top_lenth = top_lenth.max()
    min_top_lenth = top_lenth.min()
#    top_lenth_add =np.random.randint(min_top_lenth, max_top_lenth, num_add  )
    top_lenth_add =np.random.rand(num_add ) * (max_top_lenth- min_top_lenth ) +min_top_lenth
    #print(top_lenth_add) 
    #arr["height"]= height_add  
    #
    #
    weight=arr["weight"]
    #weight= arr
    max_weight = weight.max()
    min_weight = weight.min()
    weight_add =np.random.randint(min_weight , max_weight, num_add  ) 
   
    a1 = {'height':height_add
        , 'mid_lenght':mid_lenght_add
        , 'top_lenth': top_lenth_add
        , 'weight': weight_add
        }
    df = DataFrame(a1 )
    #print(df.head() )
    return df
    #"height","mid_lenght","top_lenth"


#
def proc_add_y(arr):
    num_add= 1000
#    weight=arr["weight"]
    weight= arr
    max_weight = weight.max()
    min_weight = weight.min()
    weight_add =np.random.randint(min_weight , max_weight, num_add  ) 
    return weight_add



