import pandas as pd
import numpy as np
from util import get_data,plot_data
from matplotlib import pyplot as plt

def author():
    return "ytang332"

def study_group():
    return "gburdell3"

# indicator 1 SMA
def indicator_sma(price_data,lookback_days):

    sma_df = price_data.copy()
    sma_df = sma_df * 0
    sma_df.iloc[:lookback_days-1, :] = np.nan



    for day in range(lookback_days-1,price_data.shape[0]):
        sma_df.iloc[day,:] = price_data.iloc[day - lookback_days + 1 : day + 1,:].sum()
        sma_df.iloc[day,:] = sma_df.iloc[day, :]/lookback_days


    return sma_df

# indicator 2 EMA
def indicator_ema(price, lookback_days):
    # calculate weight
    alpha = 2 / (lookback_days + 1)
    ema_df = price.copy()

    # give more weight to most recent date
    for day in range(1, price.shape[0]):
        ema_df.iloc[day, :] = alpha * price.iloc[day, :] + (1 - alpha) * ema_df.iloc[day - 1, :]
    return ema_df


# indicator 3 RSI
def indicator_rsi(price,lookback_days):

    rsi_df = price.copy()
    rsi_df = rsi_df * 0

    # get daily return
    daily_returns = price.diff(1)

    # set first day's return as nan
    for day in range(price.shape[0]):
        up_gain = daily_returns.iloc[day - lookback_days + 1 : day + 1,:].where(daily_returns >= 0).sum()
        down_loss = -1*daily_returns.iloc[day - lookback_days + 1 : day + 1,:].where(daily_returns < 0).sum()

        rs = (up_gain /lookback_days) / (down_loss /lookback_days)
        rsi_df.iloc[day,:] = 100 - (100 /(1 + rs))


    return rsi_df



#indicator 4 BBP
def indicator_bollinger(price,lookback_days):

    sma_df = indicator_sma(price,lookback_days)
    std_df = price.rolling(window=lookback_days,min_periods=lookback_days).std()
    bollinger_upper_band = sma_df + (2 * std_df)
    bollinger_buttom_band = sma_df - (2 * std_df)


    # %B (Percent B) Indicator.
    bbp = (price - bollinger_buttom_band)/(bollinger_upper_band - bollinger_buttom_band)

    return bbp

#indicator 5 MACD Histogram

def indicator_macd(price):


    ema_12 = indicator_ema(price,lookback_days=12)
    ema_26 = indicator_ema(price,lookback_days=26)

    # MACD line & MACD Single
    macd_df = ema_12 - ema_26
    macd_single =indicator_ema(macd_df,lookback_days=9)

    macd_hitogram = macd_df - macd_single

    return macd_hitogram


def run():

    sd = '2008-01-01'
    ed = '2009-12-31'
    date_range = pd.date_range(start= sd, end=ed)
    symb = ['JPM']
    price_all = get_data(symb, date_range).ffill().bfill()
    price = price_all.iloc[:,1:]

    # benchmark price
    price_all = get_data(['JPM'], date_range)
    bm_df = price_all['JPM']


    # figure 1 SMA
    plt.figure(figsize=(10,6))
    plt.title(label=f"Normalized Stock Price vs. Simple Moving Average (SMA) Indicator")

    # short term vs. long term.
    sma_12_df = indicator_sma(price, lookback_days=12)
    sma_50_df = indicator_sma(price, lookback_days=50)


    plt.plot(sma_12_df, color='darkorange')
    plt.plot(sma_50_df, color='lime')
    plt.plot(bm_df,color='purple')
    plt.xlabel('Date')
    plt.ylabel('Price & SMA')
    plt.xticks(rotation=45)
    plt.grid()

    plt.legend(["short-term: sma 12","long-term: sma 50","Benchmark"],loc='best')
    plt.savefig("./Figure_1_sma.png")



    # figure 2 EMA

    plt.figure(figsize=(10,6))

    plt.title(label=f"Normalized Stock Price vs. Exponential Moving Average (EMA) Indicator")

    # short term vs. long term.
    ema_14_df = indicator_ema(price, lookback_days=14)
    ema_50_df = indicator_ema(price, lookback_days=50)

    plt.plot(ema_14_df, color='darkorange')
    plt.plot(ema_50_df, color='lime')
    plt.plot(bm_df,color='purple')
    plt.xlabel('Date')
    plt.ylabel('Price & EMA')
    plt.xticks(rotation=45)
    plt.grid()

    plt.legend(["short-term: ema 14","long-term: ema 50","Benchmark"],loc='best')
    plt.savefig("./Figure_2_ema.png")


    # figure 3 rsi

    plt.figure(figsize=(10, 6))

    plt.title(label=f"Normalized Stock Price vs. Relative Strength Index (RSI) Indicator")

    rsi_df = indicator_rsi(price, lookback_days=14)

    plt.plot(rsi_df, color='darkorange')
    plt.plot(bm_df, color='purple')
    plt.axhline(y=30, color='r', linestyle='-', linewidth=2)
    plt.axhline(y=70, color='r', linestyle='-', linewidth=2)

    plt.xlabel('Date')
    plt.ylabel('Price & RSI')
    plt.xticks(rotation=45)
    plt.grid()

    plt.legend(["RSI",  "Benchmark"], loc='best')
    plt.savefig("./Figure_3_rsi.png")

    # figure 4 bollinger %B

    plt.figure(figsize=(14, 9))
    bbp_df = indicator_bollinger(price,lookback_days=12)

    plt.subplot(2,1,1)
    sma_df = indicator_sma(price, lookback_days=12)
    std_df = price.rolling(window=12, min_periods=12).std()
    bollinger_upper_band = sma_df + (2 * std_df)
    bollinger_buttom_band = sma_df - (2 * std_df)
    plt.plot(bollinger_upper_band, color='green')
    plt.plot(bollinger_buttom_band, color='green')
    plt.plot(bm_df, color='purple')
    plt.xlabel('Date')
    plt.ylabel('Price & Bollinger Bnad ')
    # plt.xticks(rotation=45)
    plt.grid()
    plt.legend(["Upper Band "," Lower Band","stock price"], loc='best')
    plt.title(label=f"Normalized Stock Price vs. Bollinger Bnad %B Indicator")
    plt.savefig("./Figure_4_bbp.png")


    plt.subplot(2,1,2)
    plt.plot(bbp_df, color='darkorange')
    plt.grid()
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='-', linewidth=2)
    plt.title(label=f"Bollinger Bnad %B Indicator")
    plt.ylabel('Bollinger Bnad %B')
    # Figure 5

    plt.figure(figsize = (14,10))
    macd_hist_df = indicator_macd(price)

    # Plot MACD Histogram Bars
    plt.subplot(2,1,1)
    plt.title(label=f"Normalized Stock Price vs. Moving average convergence/divergence (MACD)")
    plt.grid()
    plt.plot(bm_df, color='purple')
    plt.legend([ "Benchmark"], loc='best')

    plt.subplot(2,1,2)
    plt.grid()
    plt.title(label="MACD vs. MACD Single vs. MACD Histogram")
    # MACD line
    ema_12 = indicator_ema(price, lookback_days=12)
    ema_26 = indicator_ema(price, lookback_days=26)
    macd_df = ema_12 - ema_26
    macd_single = indicator_ema(macd_df, lookback_days=9)
    plt.plot(macd_df, color='orange')
    plt.plot(macd_single, color='skyblue')
    plt.plot(macd_hist_df, color='green')

    plt.legend(["MACD","Single Line","MACD Histogram"], loc='best')
    plt.axhline(y=0, color='r', linestyle='-', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Price & MACD')
    plt.savefig("./Figure_5_macd.png")


if __name__ == "__main__":
    pass