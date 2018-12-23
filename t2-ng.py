# -*- coding: utf-8 -*-
# DL, pred_2-ML の, DL 版(体重の予測)
# 標準ん化処理、追加　＞ＮＧ
# train/学習処理。結果ファイル保存。
# TwoLayerNet を参考に、３層ネットワーク利用
#  学習　>パラメータ保存

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# 標準化のためのモジュール
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from simple_net import SimpleNet
from util_dt import *
import time

#
# 学習データ
wdata = pd.read_csv("dat_weight.csv" 
        ,names=("weight", "height","mid_lenght","top_lenth") )
#
# 説明変数に "xx" 以外を利用
X = wdata.drop("weight", axis=1)
print(X.shape )

# 目的変数
Y = wdata["weight"]
# 学習データとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25 ,random_state=0)
#print(X_test.shape , y_test.shape  )
x_train =np.array(x_train, dtype = np.float32).reshape(len(x_train), 3)
y_train =np.array(y_train, dtype = np.float32).reshape(len(y_train), 1)
x_test  =np.array(x_test, dtype  = np.float32).reshape(len(x_test), 3)
y_test =np.array(y_test, dtype   = np.float32).reshape(len(y_test), 1)

# standard
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std  = sc.transform(x_test)

#print( x_train.shape , y_train.shape  )
print( x_train_std.shape , y_train.shape  )
print( x_test_std.shape  , y_test.shape  )
print(x_train_std[: 10])
#print(type(x_train ))
#quit()

#quit()
#
#network = SimpleNet(input_size=3 , hidden_size=10, output_size=1 )
network = SimpleNet(input_size=3 , hidden_size=2, output_size=1 )

#iters_num = 30000  # 繰り返しの回数を適宜設定する    
iters_num = 30000  # 繰り返しの回数を適宜設定する    

train_size = x_train.shape[0]
print( train_size )
#quit()

#
global_start_time = time.time()

#    batch_size = 100
#batch_size = 32
batch_size = 32

learning_rate = 0.1
#learning_rate = 0.01

train_loss_list = []
train_acc_list = []
test_acc_list = []
#
#iter_per_epoch =200
iter_per_epoch = 1000

#print(iter_per_epoch)
#quit()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    #print(batch_mask )
    x_batch = x_train_std[batch_mask]
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
#        test_acc  = network.accuracy(x_test, y_test)
        test_acc  = 0
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("i=" +str(i) + ", train acc, test acc | " + str(train_acc) + ", " + str(test_acc) + " , loss=" +str(loss) )
        print ('time : ', time.time() - global_start_time)
#pred
#train_acc = network.accuracy(x_train, y_train)
#test_acc  = network.accuracy(x_test, y_test)
#
print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc) + " , loss=" +str(loss) )
print ('time : ', time.time() - global_start_time)
#
# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")
