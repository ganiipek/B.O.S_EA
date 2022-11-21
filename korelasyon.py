import ta
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def getIndicator(dataFrame, dropna=False, fillna=False, save_csv=False, save_excel=False):
    dataFrame= ta.add_all_ta_features(dataFrame, open="open", high="high", low="low",
                                                    close="close", volume="volume", fillna=fillna)

    # dataFrame[stockSymbol] = dataFrame[stockSymbol][[
    #     "Close",
    #     "High",
    #     "Low",
    #     "Open",

    #     "others_cr",
    #     "momentum_kama",

    #     "trend_ema_slow",
    #     "trend_ema_fast",
    #     "trend_sma_slow",
    #     "trend_sma_fast",
    #     "trend_ichimoku_a",
    #     "trend_ichimoku_b",
    #     "trend_ichimoku_base",
    #     "trend_ichimoku_conv",
    #     "trend_visual_ichimoku_b",

    #     "volatility_kcc",
    #     "volatility_kch",
    #     "volatility_kcl",
    #     "volatility_dcm",
    #     "volatility_bbm",
    #     "volatility_dch",
    #     "volatility_bbh",
    #     "volatility_dcl",
    #     "volatility_bbl",

    #     "volume_vwap",
    #     "volume_obv",
    #     "volume_adi"
    # ]]
    if dropna:
        dataFrame = ta.utils.dropna(dataFrame)

    # dataFrame = dataFrame.tail(dataFrame.shape[0] - 50)

    if save_csv:
        path = "./data/korelasyon.csv"
        dataFrame.to_csv(path)
    if save_excel:
        excel = dataFrame
        excel.index = excel.index.astype(str).str[:-6]
        excel.to_excel("./data/korelasyon.xlsx")


    return dataFrame

if __name__ == "__main__":
    DATA_PATH = f"./data/dax.csv"
    data_df = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=["datetime"])
    data_df = data_df[[
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]]
    data_df = data_df.dropna()
    print(data_df)

    data_df = getIndicator(dataFrame=data_df, dropna=False, fillna=True, save_csv=True, save_excel=False)
    print(data_df)


    corrmat = data_df.corr()
    corrmat.to_excel("./data/korelasyon.xlsx")
    #saleprice correlation matrix
    k = 90 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'close')['close'].index
    cm = np.corrcoef(data_df[cols].values.T)
    sns.set(font_scale=1)
    cm = np.around(cm, decimals=6)
    hm = sns.heatmap(cm,
                    cbar=True,
                    annot=True,
                    square=True,
                    fmt='.6f',
                    annot_kws={'size': 10},
                    yticklabels=cols.values,
                    xticklabels=cols.values)
    plt.show()