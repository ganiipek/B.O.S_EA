from datetime import datetime, timedelta
import os
import sys
import ta
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import Callback
from keras import backend as K
from tensorflow import keras
from sklearn.decomposition import PCA

def getData(DATA_PATH:str):
    df_ = pd.read_csv(DATA_PATH, index_col="Local time", parse_dates=["Local time"])

    return df_


def getIndicator(df_, symbol:str, dropna=False, fillna=False, save_csv=False, save_excel=False):
    df_ = ta.add_all_ta_features(df_, open="Open", high="High", low="Low",
                                                    close="Close", volume="Volume", fillna=fillna)

    df_ = df_[[
        "Close",
        "High",
        "Low",
        "Open",

        "volatility_bbm",
        "volatility_bbh",
        "volatility_bbl",

        "volatility_kcc",
        "volatility_kch",
        "volatility_kcl",

        "volatility_dcl",
        "volatility_dch",
        "volatility_dcm",

        "trend_sma_fast",
        "trend_sma_slow",

        "trend_ema_fast",
        "trend_ema_slow",

        "trend_ichimoku_conv",
        "trend_ichimoku_base",
        "trend_ichimoku_a",
        "trend_ichimoku_b",
        "trend_visual_ichimoku_a",
        "trend_visual_ichimoku_b",

        "trend_psar_up",
        "trend_psar_down"
    ]]

    df_ = df_.tail(df_.shape[0] - 50)
    df_ = df_.fillna(0)

    if dropna:
        df_ = ta.utils.dropna(df_)

    if save_csv:
        path = "./data/indi_" + symbol + ".csv"
        df_.to_csv(path)

    if save_excel:
        excel = df_
        excel.index = excel.index.astype(str).str[:-6]
        path = "./data/indi_" + symbol + ".xlsx"
        excel.to_excel(path)

    return df_


def eksik_deger_tablosu(df):
    missing_value = df.isnull().sum()
    eksik_deger_yuzde = 100 * df.isna().sum() / len(df)
    eksik_deger_tablo = pd.concat([missing_value, eksik_deger_yuzde], axis=1)
    eksik_deger_tablo_son = eksik_deger_tablo.rename(
        columns={0: 'Eksik Değerler', 1: '% Değeri'})
    return eksik_deger_tablo_son


def split_sequence(seq, n_steps_in, n_steps_out):
    # Creating a list for both variables
    data_X, data_Y = [], []

    for i in range(len(seq)):
        end = i + n_steps_in
        out_end = end + n_steps_out

        if out_end > len(seq):
            break

        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        seq_x, seq_y = seq[i:end, :], seq[end:out_end,0]
        #print("seq_x", seq_x)
        #print("seq_y", seq_y)
        data_X.append(seq_x)
        data_Y.append(seq_y)

    return np.array(data_X), np.array(data_Y)


def pick_bilme_orani(decision_values, real_values):
    dogru_tahmin = 0
    yanlis_tahmin = 0
    for index, (decision, real) in enumerate(zip(decision_values, real_values)):
        if index < len(decision_values) - 1:
            if decision.astype(str) != "nan" and decision_values[index + 1].astype(str) != "nan":
                if decision < decision_values[index + 1] and real < real_values[index + 1]:
                    dogru_tahmin = dogru_tahmin + 1
                elif decision > decision_values[index + 1] and real > real_values[index + 1]:
                    dogru_tahmin = dogru_tahmin + 1
                else:
                    yanlis_tahmin = yanlis_tahmin + 1

    return dogru_tahmin / (dogru_tahmin + yanlis_tahmin) * 100


def validater(symbol, df, model, n_features, close_scaler, n_per_in, n_per_out, intervall, grafik=False):
    # Creating an empty DF to store the predictions
    predictions = pd.DataFrame(index=df.index, columns=[df.columns[0]])
    # print("kontrol1",predictions)

    for i in range(n_per_out, len(df) - n_per_in, n_per_out):
        #print("---------------")
        #print(i,"-",range(n_per_out, len(df) - n_per_in, n_per_out))
        # Creating rolling intervals to predict off of

        x = df[-i - n_per_in:-i]
        # print(-i - n_per_in,":",-i)
        # Predicting using rolling intervals
        yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))
        # Transforming values back to their normal prices
        yhat = close_scaler.inverse_transform(yhat)[0]
        # DF to store the values and append later, frequency uses business days

        if intervall == "1m":
            pred_df = pd.DataFrame(yhat,
                                   index=pd.date_range(start=x.index[-1] + timedelta(minutes=1),
                                                       periods=len(yhat),
                                                       freq="min"),
                                   columns=[x.columns[0]])
        elif intervall == "daily":
            pred_df = pd.DataFrame(yhat,
                                   index=pd.date_range(start=x.index[-1] + timedelta(days=1),
                                                       periods=len(yhat),
                                                       freq="B"),
                                   columns=[x.columns[0]])
        # Updating the predictions DF

        predictions.update(pred_df)

        percent = round((i / (len(df) - n_per_in)) * 100, 2)
        progress_string = f"\rPredict: {i}/{len(df) - n_per_in} | %{percent}"
        sys.stdout.write(progress_string)
        sys.stdout.flush()


    # predictions = predictions[-(len(df) - n_per_in):]
    # predictions = predictions.fillna(method="ffill")
    actual = pd.DataFrame(close_scaler.inverse_transform(df[["Close"]]), index= df.index, columns=[x.columns[0]])
    # actual = actual[-(len(df) - n_per_in):]

    #print("Predict Boyutu", predictions.shape, "NaN Sayısı:", predictions.isna().sum().sum())
    model_rmse =  val_rmse(actual, predictions)
    pick_bilme_oranı = 0 #pick_bilme_orani(predictions.values, actual.values)

    if grafik:
        plt.figure(figsize=(16, 6))
        plt.plot(actual, label="Actual Prices")
        plt.plot(predictions, label="Predicted Prices")
        plt.ylabel("Close")
        plt.xlabel("Dates")
        plt.legend()
        plt.show()

    return predictions, actual, model_rmse, pick_bilme_oranı


def simulate_trade(df_actual, df_predict, n_per_in, n_per_out, fee=0, detail=False):
    eski_fiyat = 0
    profit = 0
    budget = 0
    total_buy, total_sell = 0, 0
    bought = False
    df_trade = pd.DataFrame(index=df_actual.index, columns=["Type", "Current Price"])

    for i in range(0, len(df_actual)-n_per_out+1):
        predict = df_predict[-i - n_per_out: -i]
        previous_actual = df_actual[-i - n_per_out - 1: -i - n_per_out]
        if not predict.empty and not previous_actual.empty and not previous_actual.isnull().any().any() and not predict.isnull().any().any():
            date = previous_actual.index
            previous_actual = round(previous_actual.values[0][0],2)
            #print("previous_actual:", previous_actual)
            #print("predict", predict)
            #print("predict['Close'].mean()", predict["Close"].mean())
            if not bought and round(predict['Close'].mean(),2) > previous_actual + previous_actual*fee: #buying
                bought = True
                eski_fiyat = previous_actual
                budget -= previous_actual
                if detail:
                    print(f"BUY: Hisse {previous_actual} fiyattan satın alındı. Toplam bütçe: {budget}")
                df_trade_local = pd.DataFrame(data={"Type":"buy", "Current Price":previous_actual}, index=date)
                df_trade.update(df_trade_local)
                total_buy +=1
            elif bought and round(predict['Close'].mean(),2) < previous_actual + previous_actual*fee: #selling
                bought = False
                profit = previous_actual - eski_fiyat - previous_actual * fee
                budget += previous_actual - previous_actual * fee
                if detail:
                    print(f"SELL: Hisse {previous_actual} fiyattan satıldı. Toplam kar: {profit}, Toplam bütçe: {budget}")
                df_trade_local = pd.DataFrame(data={"Type": "sell", "Current Price":previous_actual}, index=date)
                df_trade.update(df_trade_local)
                total_sell += 1
    df_trade = df_trade.dropna()

    return df_trade, budget, total_buy, total_sell


def compare_trades(y_true, y_pred, n_per_in, n_per_out, fee=0, grafik=False, detail=False):
    df_trade, profit, total_buy, total_sell = simulate_trade(y_true, y_pred, n_per_in, n_per_out, fee=fee,detail=detail)

    #mape = mean_absolute_percentage_error(y_true_profits, y_pred_profits)
    if grafik:
        plt.figure(figsize=(14, 6))
        plt.title(f"Money Difference. It's to compare the points of transactions")
        plt.scatter(df_trade.loc[df_trade['Type'] == "sell"].index, df_trade.loc[df_trade['Type'] == "sell", ["Current Price"]], color="green", label="Sell", zorder=10)
        plt.scatter(df_trade.loc[df_trade['Type'] == "buy"].index, df_trade.loc[df_trade['Type'] == "buy", ["Current Price"]], color="red", label="Buy", zorder=10)
        plt.plot(y_true, label="Actual Prices",zorder=5)
        plt.plot(y_pred, label="Predicted Prices",zorder=0)
        plt.legend()
        plt.show()

    return profit, total_buy, total_sell


def mean_absolute_percentage_error(actual, pred):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    actual, pred = np.array(actual), np.array(pred)
    actual = np.where(actual == 0, 1, actual)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def val_rmse(df1, df2):
    """
    Calculates the root mean square error between the two Dataframes
    """
    df = df1.copy()

    # Adding a new column with the closing prices from the second DF
    df['close2'] = df2.Close

    # Dropping the NaN values
    df.dropna(inplace=True)

    # Adding another column containing the difference between the two DFs' closing prices
    df['diff'] = df.Close - df.close2

    # Squaring the difference and getting the mean
    rms = (df[['diff']] ** 2).mean()

    # Returning the sqaure root of the root mean square
    return float(np.sqrt(rms))


def forecastingFuture(df, model, close_scaler, n_per_in, n_features):
    # Predicting off of the most recent days from the original DF
    yhat = model.predict(np.array(df.tail(n_per_in)).reshape(1, n_per_in, n_features))

    # Transforming the predicted values back to their original format
    yhat = close_scaler.inverse_transform(yhat)[0]

    # Creating a DF of the predicted prices
    preds = pd.DataFrame(yhat,
                         index=pd.date_range(start=df.index[-1] + timedelta(days=1),
                                             periods=len(yhat),
                                             freq="B"),
                         columns=[df.columns[0]])

    # Number of periods back to plot the actual values
    pers = n_per_in

    # Transforming the actual values to their original price
    actual = pd.DataFrame(close_scaler.inverse_transform(df[["Close"]].tail(pers)),
                          index=df.Close.tail(pers).index,
                          columns=[df.columns[0]]).append(preds.head(1))

    # Printing the predicted prices
    print(preds)

    # Plotting
    plt.figure(figsize=(16, 6))
    plt.plot(actual, label="Actual Prices")
    plt.plot(preds, label="Predicted Prices")
    plt.ylabel("Price")
    plt.xlabel("Dates")
    plt.title(f"Forecasting the next {len(yhat)} days")
    plt.legend()
    plt.show()

