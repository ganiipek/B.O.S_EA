# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 18:21:19 2022

@author: gani.ipek
"""
# %% Import
import functions
import train_LSTM
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from datetime import datetime

# %% FOR REPRODUCIBILITY
np.random.seed(7)
print(f"Tensorflow Version: {tf.__version__}")
print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)

# %% SETTINGS
symbol = "EURUSD"

n_per_in = 128
n_per_out = 1  # fonksiyonlar.validater freq değiştirmeyi unutma şu anda = 1min

grafik = False
train = False
intervall = "hourly" # 1m

activation_function = "tanh" #relu
activation_function_last = "tanh" #tanh
optimizer = "adam" # adagrad, adam, RMSprop
epoch_size = 10
batch_size = 32
validation_split = 0.10
shuffle = True

# %% Loading Data
data = "indi_EURUSD_2013-01-01 00_00_00_2022-01-02 00_00_00_1hour"

df_ = functions.getData(f"./data/{data}.csv")
# df_ = functions.getData(f"./data/indi_DAX.csv")
# df_["Close"] = df_["Close"] - df_["Close"].shift(1)

# df_ = functions.getIndicator(df_=df_, symbol=symbol, data=data, dropna=False, fillna=False, save_csv=True, save_excel=False)
print(f"Data Shape: {df_.shape}")
# df_ = df_.tail(100000)
# print(functions.eksik_deger_tablosu(df_))

# %% Preprocessing Data
close_scaler = MinMaxScaler(feature_range=(0,1))
close_scaler.fit(df_[['Close']])

scaler = MinMaxScaler(feature_range=(0,1))
df_ = pd.DataFrame(scaler.fit_transform(df_), columns=df_.columns, index=df_.index)

n_features = df_.shape[1]

if train:
    x_data, y_data = functions.split_sequence(df_.to_numpy(), n_per_in, n_per_out)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=validation_split)

    print("\n\n\n")
    print("X_train -> ", X_train.shape, "y_train -> ", y_train.shape)
    print("X_test -> ", X_test.shape, "y_test -> ", y_test.shape)
    print("\n\n\n")

    model = train_LSTM.dailyTrainLSTM(
        X_train,
        X_test,
        y_train, 
        y_test, 
        epoch_size,
        batch_size,
        shuffle,
        n_per_in, 
        n_per_out, 
        n_features, 
        grafik=True
        )

    # model.save(os.path.dirname(os.path.dirname(__file__)) + "/model/" + symbol + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M") + ".h5")
    model.save(f"./Model/{symbol}_{intervall}.h5")
    print("Saved model to disk\n")
else:
    model = keras.models.load_model(f"./Model/{symbol}_{intervall}.h5")
    #model.summary()
    print("Old model loaded\n")

# %% Predict
predictions, actual, model_rmse, pick_bilme_oranı = functions.validater(symbol, df_.tail(500), model, n_features, close_scaler, n_per_in, n_per_out, intervall, grafik=True)

# profit, total_buy, total_sell = functions.compare_trades(actual, predictions, n_per_in, n_per_out, fee=0, grafik=False, detail=False)

# model_statistic = {"symbol": symbol,
#                     "total_buy": total_buy,
#                     "total_sell": total_sell,
#                     "profit": profit,
#                     "model_rmse": model_rmse,
#                     "pick_bilme_oranı": pick_bilme_oranı}

# print(f"\n\nRMSE: {model_rmse} , Pick Bilme Oranı: %{pick_bilme_oranı}")
# print(f"Alım işlemi: {model_statistic['total_buy']} adet, Satım işlemi: {model_statistic['total_sell']} adet -> Kar: {model_statistic['profit']}")


# model_statistics = []
# for symbol in firma_array:
#     print("Şirket: " + symbol + ", dataFrame Boyutu: ", df_.shape)
#     #print(functions.eksik_deger_tablosu(df_))

#     close_scaler = MinMaxScaler(feature_range=(0,1))
#     close_scaler.fit(df_[['Close']])

#     scaler = MinMaxScaler(feature_range=(0,1))
#     df_ = pd.DataFrame(scaler.fit_transform(df_), columns=df_.columns, index=df_.index)

#     df_basic_data_ = df_[["Close","High","Low","Open"]]
#     df_ = df_.drop(columns = ["Close","High","Low","Open"])

#     pca = PCA(n_components=10)
#     df_ = pd.DataFrame(pca.fit_transform(df_), index=df_.index)

#     df_basic_data_ = df_basic_data_.join(df_)
#     df_ = df_basic_data_

#     n_features = df_.shape[1]
#     print("n_features",n_features)
#     if train:
#         x_data, y_data = functions.split_sequence(df_.to_numpy(), n_per_in, n_per_out)
#         X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=validation_split)
#         print("X_train -> ", X_train.shape, "y_train -> ", y_train.shape)
#         print("X_test -> ", X_test.shape, "y_test -> ", y_test.shape)

#         if 'model' in vars():
#             del model, close_scaler, scaler, predictions, actual, model_rmse, pick_bilme_oranı
#             print("Model refresh")

#         model = train_LSTM.dailyTrainLSTM(X_train, X_test, y_train, y_test, n_per_in, n_per_out, n_features, grafik=True)

#         # model.save(os.path.dirname(os.path.dirname(__file__)) + "/model/" + symbol + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M") + ".h5")
#         model.save(f"{os.path.dirname(os.path.dirname(__file__))}/model/{symbol}_{intervall}.h5")
#         print("Saved model to disk\n")
#     else:
#         model = keras.models.load_model(f"{os.path.dirname(os.path.dirname(__file__))}/model/{symbol}_{intervall}.h5")
#         #model.summary()
#         print("Old model loaded\n")

#     actual = {}
#     predictions = {}
#     mean_kar = 0
#     mean_pick = 0

#     predictions, actual, model_rmse, pick_bilme_oranı = functions.validater(symbol, df_, model, n_features, close_scaler, n_per_in, n_per_out, intervall, grafik=True)

#     profit, total_buy, total_sell = functions.compare_trades(actual, predictions, n_per_in, n_per_out, fee=0, grafik=False, detail=False)

#     model_statistic = {"symbol": symbol,
#                        "total_buy": total_buy,
#                        "total_sell": total_sell,
#                        "profit": profit,
#                        "model_rmse": model_rmse,
#                        "pick_bilme_oranı": pick_bilme_oranı}

#     model_statistics.append(model_statistic)

#     del model

# for model_statistic in model_statistics:
#     print(f"\nŞirket: {model_statistic['symbol']}")
#     print(f"RMSE: {model_statistic['model_rmse']} , Pick Bilme Oranı: %{model_statistic['pick_bilme_oranı']}")
#     print(f"Alım işlemi: {model_statistic['total_buy']} adet, Satım işlemi: {model_statistic['total_sell']} adet -> Kar: {model_statistic['profit']}")

"""
mean_kar = mean_kar + (y_pred_profits / y_true_profits * 100)
mean_pick = mean_pick + pick_bilme_oranı
print("\nToplam şirket sayısı:", len(stockList))
print("Ortalama Pick noktası tahmin oranı: %", mean_pick / len(stockList))
print("Ortalama Kar tahmin oranı: %", mean_kar / len(stockList))
"""