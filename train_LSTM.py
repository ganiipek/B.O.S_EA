import functions
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from datetime import datetime

def example():
    pass

def dailyTrainLSTM(X_train, X_test, y_train, y_test, n_per_in, n_per_out, n_features, grafik=False):
    optimizer = Adam()

    earlyCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    log_dir = f"./Logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = keras.Sequential()
    model.add(layers.LSTM(64, activation="tanh", return_sequences=True, input_shape=(n_per_in, n_features)))
    model.add(layers.Dropout(0.1))
    model.add(layers.LSTM(32, activation="tanh", return_sequences=True))
    model.add(layers.Dropout(0.1))
    model.add(layers.LSTM(16, activation="tanh", return_sequences=True))
    model.add(layers.Dropout(0.1))
    model.add(layers.LSTM(8, activation="tanh"))
    model.add(layers.Dense(n_per_out))
    model.summary()
    # print(model.get_config())
    model.compile(optimizer=optimizer, loss="mae", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model_grafik = model.fit(X_train, y_train,
                             validation_data=(X_test, y_test),
                             epochs=100,
                             batch_size=128,
                             shuffle=True)
                            #  callbacks=[earlyCallback])

    if grafik:
        plt.figure(figsize=(14, 3))
        plt.subplot(1, 2, 1)
        plt.plot(model_grafik.history['root_mean_squared_error'])
        plt.plot(model_grafik.history['val_root_mean_squared_error'])
        plt.title('Model root_mean_squared_error')
        plt.ylabel('root_mean_squared_error')
        plt.xlabel('Epoch')
        plt.legend(['root_mean_squared_error', 'val_root_mean_squared_error'], loc='upper left')
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(model_grafik.history['loss'])
        plt.plot(model_grafik.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.show()

    return model


def lstm2(X_train, X_test, y_train, y_test, n_per_in, n_per_out, n_features, grafik=False):
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(64, activation='relu'), input_shape=(n_per_in, n_features)))
    model.add(layers.Dense(n_per_out))
    model.compile(optimizer='adam', loss='mse', metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

    earlyCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model_grafik = model.fit(X_train, y_train,
                             validation_data=(X_test, y_test),
                             epochs=100,
                             batch_size=32,
                             shuffle=True,
                             callbacks=[earlyCallback])
    if grafik:
        plt.figure(figsize=(14, 3))
        plt.subplot(1, 2, 1)
        plt.plot(model_grafik.history['mae'])
        plt.plot(model_grafik.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['mae', 'val_mae'], loc='upper left')
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(model_grafik.history['loss'])
        plt.plot(model_grafik.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.show()

    return model