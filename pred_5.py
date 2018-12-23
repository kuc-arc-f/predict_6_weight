# -*- coding: utf-8 -*-
# 評価
#

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from simple_net import SimpleNet
from util_dt import *
import time
import pickle

#
if __name__ == '__main__':
    # 学習データ
    global_start_time = time.time()
    # 学習データ
    wdata = pd.read_csv("dat_weight_4.csv" )

    #
    # 説明変数に "xx" 以外を利用
    X = wdata.drop("weight", axis=1)
    X = X.drop("index", axis=1)
    print(X.head() )
    num_max_y= 1000
    X = (X / num_max_y )
    print(X.shape )
    #quit()

    # 目的変数
    Y = wdata["weight"]
    Y = Y / num_max_y

    # 学習データとテストデータに分ける
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25 ,random_state=0)
    #print(X_test.shape , y_test.shape  )
    x_train =np.array(x_train, dtype = np.float32).reshape(len(x_train), 3)
    y_train =np.array(y_train, dtype = np.float32).reshape(len(y_train), 1)
    x_test  =np.array(x_test, dtype  = np.float32).reshape(len(x_test), 3)
    y_test =np.array(y_test, dtype   = np.float32).reshape(len(y_test), 1)

    print( x_train.shape , y_train.shape  )
    print( x_test.shape  , y_test.shape  )
    print(x_train[: 10])
    #quit()

    # load model
#    network = SimpleNet(input_size=1 , hidden_size=10, output_size=1 )
    network = SimpleNet(input_size=3 , hidden_size=10, output_size=1 )
    network.load_params("params.pkl" )
    #print( network.params["W1"] )
    #pred
    train_acc = network.accuracy(x_train, y_train)
    test_acc  = network.accuracy(x_test, y_test)
    #
    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc)   )
    #
#    x_test_dt= conv_num_date(x_test_pred )
#    x_train_dt= conv_num_date(x_train )
    #print(x_test_dt.shape )
    print(x_test[: 10]  )
    y_val = network.predict(x_test[: 10])
    y_val = y_val * num_max_y
    print(y_val )
#    quit()

    y_train = y_train * num_max_y
    y_val   = y_val * num_max_y    
    print ('time : ', time.time() - global_start_time)
    #print(y_val[:10] )
    #print(x_test_dt[:10] )
    quit()
    #plt
    plt.plot(x_train_dt, y_train, label = "temp")
    plt.plot(x_test_dt , y_val , label = "predict")
    plt.legend()
    plt.grid(True)
    plt.title("weight pred")
    plt.xlabel("x")
    plt.ylabel("temperature")
    plt.show()
