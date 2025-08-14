import numpy as np
import pandas as pd
from util import get_data,plot_data
import datetime as dt
import indicators as ind
import marketsimcode as msi
import matplotlib.pyplot as plt

"""
Indicators:

indicator_rsi
indicator_bollinger
indicator_macd

"""



class ManualStrategy:

    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission




    def testPolicy(self,symbol="JPM",
                   sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009,12,31),
                   sv = 100000):

        # initialize getting date range and price
        # pulling more dates for indicator calculation
        sd = sd - dt.timedelta(days=30)


        date_range = pd.date_range(start=sd, end=ed)
        prices_all = get_data([symbol], date_range).bfill().ffill()
        prices = prices_all.iloc[:,1:]

        # reset date back to requested start date
        sd = dt.datetime(2008, 1, 1)
        ind_macd = ind.indicator_macd(prices)
        ind_macd = ind_macd[ind_macd.index >=sd]
        # print(ind_macd)
        ind_rsi = ind.indicator_rsi(prices,lookback_days=12)
        ind_rsi = ind_rsi[ind_rsi.index >=sd]
        ind_bollinger = ind.indicator_bollinger(prices,lookback_days=12)
        ind_bollinger = ind_bollinger[ind_bollinger.index >=sd]


        prices = prices[prices.index >= sd]

        df_trades = prices.copy() * 0
        df_orders = prices.copy() * 0
        df_index = prices.index


        # tracking the position change
        current_position = 0
        for sym in [symbol]:
            for day in range(1,prices.index.shape[0] - 1):
                # print(prices.loc[df_index[day],sym])
                if (sum([(#(ind_macd.loc[df_index[day],sym] > 0.0) and
                         (ind_macd.loc[df_index[day],sym] > 0)) ,

                          (ind_rsi.loc[df_index[day],sym] < 40),
                         (ind_bollinger.loc[df_index[day],sym] < 0.3)] ) == 3 ) :
                    # long
                    if current_position == 0:
                        df_trades.loc[df_index[day], sym] = 1000
                        current_position = 1000
                        df_orders.loc[df_index[day], sym] = 1

                    if current_position ==  -1000:
                        df_trades.loc[df_index[day], sym] = 2000
                        current_position += 2000
                        df_orders.loc[df_index[day], sym] = 1

                elif  sum([(#(ind_macd.loc[df_index[day],sym] < 0.0) and
                           (ind_macd.loc[df_index[day-1],sym] < 0)) ,
                           (ind_rsi.loc[df_index[day],sym] > 60),
                           (ind_bollinger.loc[df_index[day],sym] > 0.7)]) == 3 :

                     # short
                    if current_position == 0:
                        df_trades.loc[df_index[day], sym] = -1000
                        current_position -= 1000
                        df_orders.loc[df_index[day], sym] = -1

                    elif current_position == 1000:
                        df_trades.loc[df_index[day], sym] = -2000
                        current_position -= 2000
                        df_orders.loc[df_index[day], sym] = -1

        return df_trades

    def benchmark(self,symbol="JPM",
                   sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009,12,31),
                  sv = 100000,
                  commission=9.95,
                  impact=0.005):

        date_range = pd.date_range(start=sd, end=ed)

        prices_all = get_data([symbol], date_range).bfill().ffill()
        prices = prices_all.iloc[:, 1:]
        date_index = prices.index
        port_columns = ['portfolio value'] + ['JPM'] + ['cash']
        portfolio_val = pd.DataFrame(0, index=date_index, columns=port_columns)
        impacts = impact * 1000 * prices.iloc[0, 0]
        portfolio_val.iloc[0,2] = sv - prices.iloc[0,0]*1000 - impacts - commission
        portfolio_val.iloc[:,2] = portfolio_val.iloc[0,2]
        portfolio_val.iloc[:,1] = prices.iloc[:,0] * 1000
        portfolio_val.iloc[:,0] = portfolio_val.sum(axis=1)


        return portfolio_val.iloc[:,[0]]



    def plot_all(self,benchmark=None,
                 strategy=None,
                 buy_date=dt.datetime(2009,12,31),
                 sell_date=dt.datetime(2009,12,31),
                 title=''):

        if benchmark is None:
            print(" No benchmark data provided.")
            return
        if strategy is None:
            print("No strategy data provided.")
            return


        benchmark = benchmark/benchmark.iloc[0]
        strategy = strategy/strategy.iloc[0]


        buy_date = pd.to_datetime(buy_date)
        sell_date = pd.to_datetime(sell_date)


        plt.figure(figsize=(16, 8))

        # plot price
        plt.plot(benchmark, label='Benchmark', color='purple')
        plt.plot(strategy, label='Manual Strategy', color='red')
        for date in buy_date:
            # print(date)
            plt.axvline(date, color='blue', linestyle='--')
        for date in sell_date:
           plt.axvline(date, color='black', linestyle='--', alpha=1)


        plt.title(f'{title} Performance: Benchmark vs. Manual Strategy ({benchmark.index[0].year}-{benchmark.index[-1].year})',fontsize=20)
        plt.xlabel('Date', fontsize=15)
        plt.ylabel('Normalized Portfolio Value', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.savefig(f"./images/ManualStrategy_{title}.png")
        pass

    def performance_metrics(self,df_portval=None):

        if df_portval is None:
            return  None

        # Cumulative return
        cul_return = df_portval.iloc[-1]/df_portval.iloc[0] - 1
        cul_return = cul_return.values[0]

        # STDEV of daily returns
        daily_return = df_portval/df_portval.shift(1) - 1
        std_daily_return = daily_return.std().values[0]

        # mean of daily retunr
        mean_daily_return = daily_return.mean().values[0]

        return cul_return, std_daily_return, mean_daily_return



    def support_plot(self,
                     symbol="JPM",
                     sd=dt.datetime(2008, 1, 1),
                     ed=dt.datetime(2009,12,31),
                     buy_date=[dt.datetime(200,12,31)],
                     sell_date=[dt.datetime(2009,12,31)],):

        sd = sd - dt.timedelta(days=30)


        date_range = pd.date_range(start=sd, end=ed)
        prices_all = get_data([symbol], date_range).bfill().ffill()
        prices = prices_all.iloc[:, 1:]


        # reset date back to requested start date
        sd = dt.datetime(2008, 1, 1)
        ind_macd = ind.indicator_macd(prices)
        ind_macd = ind_macd[ind_macd.index >= sd]
        ind_rsi = ind.indicator_rsi(prices, lookback_days=12)
        ind_rsi = ind_rsi[ind_rsi.index >= sd]
        ind_bollinger = ind.indicator_bollinger(prices, lookback_days=12)
        ind_bollinger = ind_bollinger[ind_bollinger.index >= sd]

        prices = prices[prices.index >= sd]

        buy_date = pd.to_datetime(buy_date)
        buy_price = prices.loc[buy_date]
        sell_price = prices.loc[sell_date]



        # MACD Histogram

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 10), sharex=True)
        plt.title("Stock Price vs. MACD vs. RSI vs. %B (2008-2009)", fontsize=20)


        axes[0].plot(prices.index, prices.iloc[:], label='Stock Price', color='blue')
        axes[0].scatter(buy_date, buy_price, color='green', label='Buy', zorder=5)
        axes[0].scatter(sell_date, sell_price, color='red', label='Sell', zorder=5)
        axes[0].set_ylabel('Price',fontsize=16)
        axes[0].legend(fontsize=14)
        axes[0].set_title("Stock Price with buy & sell point (2008-2009)",fontsize=18)
        axes[0].grid(True)

        # MACD Histogram
        normalize_price = prices.iloc[:]/prices.iloc[0]
        colors = ['green' if val > 0 else 'red' for val in ind_macd.iloc[:, 0]]
        axes[1].bar(ind_macd.index, ind_macd.iloc[:,0], label='MACD Histogram', color=colors)
        axes[1].plot(normalize_price.index, normalize_price.iloc[:, 0], label='Normalized price', color='blue')
        axes[1].set_ylabel('MACD',fontsize=16)
        axes[1].set_title("MACD Histogram",fontsize=18)
        axes[1].legend(fontsize=14)
        axes[1].grid(True)

        # RSI
        axes[2].plot(ind_rsi.index, ind_rsi.iloc[:,0], label='RSI', color='orange')
        axes[2].axhline(70, linestyle='--', color='red')
        axes[2].axhline(30, linestyle='--', color='green')
        axes[2].set_ylabel('RSI',fontsize=16)
        axes[2].legend(fontsize=14)
        axes[2].set_title("RSI",fontsize=18)
        axes[2].grid(True)

        # %B
        axes[3].plot(ind_bollinger.index, ind_bollinger.iloc[:,0], label='%B', color='purple')
        axes[3].axhline(1, linestyle='--', color='gray')
        axes[3].axhline(0, linestyle='--', color='gray')
        axes[3].axhline(0.7, linestyle='--', color='gray')
        axes[3].axhline(0.3, linestyle='--', color='gray')
        axes[3].set_ylabel('%B',fontsize=16)
        axes[3].set_title("Percent B",fontsize=18)
        prices = prices/prices.iloc[0, 0]
        plt.plot(prices, color='dodgerblue',label='Price')
        plt.plot(ind_macd,color='orange',label='MACD')
        axes[3].legend(fontsize=14)
        axes[3].grid(True)
        plt.savefig(f"./images/ManualStrategy_support_plot.png")






    def author(self):
        return "ytang332"

    def study_group(self):
        return "gburdell3"


if __name__ == "__main__":



    strategy = ManualStrategy(verbose=False)
    symbol = 'JPM'

    #----------in-sample period----------

    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    df_trades = strategy.testPolicy(symbol=symbol,
                                    sd=sd_in,
                                    ed=ed_in,
                                    sv = sv,)
    sell_date_in = df_trades[df_trades[[symbol][0]] < 0].index
    buy_date_in = df_trades[df_trades[[symbol][0]] > 0].index


    # calculate portofolio value
    portvals_in = msi.compute_portvals(df_trades,start_val=sv,commission=commission, impact=impact)

    # benchmark
    bm_df_in = strategy.benchmark(symbol=symbol,
                               sd=sd_in,
                               ed=ed_in,
                               sv = sv,
                               commission=commission,
                               impact=impact)

    # plot
    strategy.plot_all(benchmark=bm_df_in,strategy=portvals_in,buy_date=buy_date_in,sell_date=sell_date_in,title='In-Sample')

    #----------out of sample period-----------#

    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)

    # Strategy
    df_trades = strategy.testPolicy(symbol=symbol,
                                    sd=sd_out,
                                    ed=ed_out,
                                    sv=sv, )
    sell_date_out = df_trades[df_trades[[symbol][0]] < 0].index
    buy_date_out = df_trades[df_trades[[symbol][0]] > 0].index


    # calculate portofolio value
    portvals_out = msi.compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)

    # Benchmark
    bm_df_out = strategy.benchmark(symbol=symbol,
                               sd=sd_out,
                               ed=ed_out,
                               sv=sv,
                               commission=commission,
                               impact=impact)

    # plot
    strategy.plot_all(benchmark=bm_df_out, strategy=portvals_out, buy_date=buy_date_out, sell_date=sell_date_out, title='Out-of-Sample')

    # ----------Performance Metrics-----------#

    cr_in_port, std_in_port, mean_in_port = strategy.performance_metrics(df_portval=portvals_in)
    cr_in_bm, std_in_bm, mean_in_bm = strategy.performance_metrics(bm_df_in)
    cr_out_port, std_out_port, mean_out_port = strategy.performance_metrics(portvals_out)
    cr_out_bm, std_out_bm, mean_out_bm = strategy.performance_metrics(bm_df_out)


    data_in = [[cr_in_port, std_in_port, mean_in_port],
            [cr_in_bm, std_in_bm, mean_in_bm],]

    performance_table_in = pd.DataFrame(data_in, columns=['Cumulative Return', 'STDEV of daily return', 'mean of daily return'],index=['Portfolio', 'Benchmark'])
    performance_table_in.to_csv('ManulStrategy_InSample_statistics.csv')

    data_out = [[cr_out_port, std_out_port, mean_out_port],
                [cr_out_bm, std_out_bm, mean_out_bm],]

    performance_table_out = pd.DataFrame(data_out,
                                     columns=['Cumulative Return', 'STDEV of daily return', 'mean of daily return'],
                                     index=['Portfolio', 'Benchmark'])
    performance_table_out.to_csv('ManulStrategy_OutofSample_statistics.csv')


    # suprrot plot:
    strategy.support_plot(symbol=symbol,sd= dt.datetime(2008, 1, 1),ed=dt.datetime(2009,12,31),buy_date=buy_date_in,sell_date=sell_date_in)