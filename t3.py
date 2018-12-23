# -*- coding: utf-8 -*-
# csv列、変更対応。
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
from util_df import *
import time

#
# 学習データ
wdata = pd.read_csv("dat_weight_2.csv" 
        ,names=("index" ,"weight", "height","mid_lenght","top_lenth") )

#print( wdata.head() )
#quit()

#
# 説明変数に "xx" 以外を利用
X = wdata.drop("weight", axis=1)
X = X.drop("index", axis=1)
#X =np.array(X, dtype = np.float32).reshape(len(X), 3)
#X_sub=X
#X_sub = X_sub.assign(height=pd.to_numeric( X_sub.height ))
#quit()


X =proc_add_arr(X)
#print(X.info() )

num_max_y= 1000
X = (X / num_max_y )
print(X.head() )
print(X.shape )
#quit()

# 目的変数
Y = wdata["weight"]
#quit()

Y= proc_add_y(Y)
#print(Y.head() )
print(type(X ) )
#quit()

Y = Y / num_max_y
print(Y.shape )


# 学習データとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25 ,random_state=0)
x_train =np.array(x_train, dtype = np.float32).reshape(len(x_train), 3)
y_train =np.array(y_train, dtype = np.float32).reshape(len(y_train), 1)
x_test  =np.array(x_test, dtype  = np.float32).reshape(len(x_test), 3)
y_test =np.array(y_test, dtype   = np.float32).reshape(len(y_test), 1)

print( x_train.shape , y_train.shape  )
print( x_test.shape  , y_test.shape  )
#print(x_train[: 10])
#print(type(x_train ))
quit()
#
network = SimpleNet(input_size=3 , hidden_size=10, output_size=1 )

#iters_num = 30000  # 繰り返しの回数を適宜設定する    
iters_num = 10000  # 繰り返しの回数を適宜設定する    

train_size = x_train.shape[0]
print( train_size )
#quit()

#
global_start_time = time.time()

#    batch_size = 100
#batch_size = 32
batch_size = 8

#learning_rate = 0.1
learning_rate = 0.01

train_loss_list = []
train_acc_list = []
test_acc_list = []
#
#iter_per_epoch =200
iter_per_epoch = 500

#print(iter_per_epoch)
#quit()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    #print(batch_mask )
    x_batch = x_train[batch_mask]
    t_batch = y_train[batch_mask]
    #quit()
    
    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc  = network.accuracy(x_test, y_test)
#        test_acc  = 0
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("i=" +str(i) + ", train acc, test acc | " + str(train_acc) + ", " + str(test_acc) + " , loss=" +str(loss) )
        print ('time : ', time.time() - global_start_time)
#pred
train_acc = network.accuracy(x_train, y_train)
test_acc  = network.accuracy(x_test, y_test)
#
print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc) + " , loss=" +str(loss) )
print ('time : ', time.time() - global_start_time)
#
# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")
