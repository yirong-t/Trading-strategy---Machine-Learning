
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from StrategyLearner import *
import ManualStrategy as ms
import marketsimcode as msi
import datetime as dt
import experiment1 as e1
import experiment2 as e2


if __name__ == "__main__":

    # initialize
    impact_1 = 0.005
    commission = 9.95
    symbol = 'JPM'
    sv = 100000

    # in-sample period
    sd_insample=dt.datetime(2008, 1, 1)
    ed_insample=dt.datetime(2009, 12, 31)

    sd_out=dt.datetime(2010, 1, 1)
    ed_out=dt.datetime(2011, 12, 31)
    #

    # experiment 1:
    e1.experiemnt1_plot(symbol,
                        sd_insample=sd_insample,
                        ed_insample=ed_insample,
                        sd_outsample=sd_out,
                        ed_outsample=ed_out,
                        sv=sv,
                        impact=impact_1,
                        commission=commission,)


    e2.experiment2(symbol=symbol,
                        impact_1 = 0.00,
                        impact_2 = 0.05,
                        impact_3 = 0.1,
                        sd = dt.datetime(2008, 1, 1),
                        ed = dt.datetime(2009, 12, 31),
                        sv=100000)


    ## Manual strategy

    strategy = ms.ManualStrategy(verbose=False)
    symbol = 'JPM'

    # ----------in-sample period----------

    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    df_trades = strategy.testPolicy(symbol=symbol,
                                    sd=sd_in,
                                    ed=ed_in,
                                    sv=sv, )
    sell_date_in = df_trades[df_trades[[symbol][0]] < 0].index
    buy_date_in = df_trades[df_trades[[symbol][0]] > 0].index


    # calculate portofolio value
    portvals_in = msi.compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)

    # benchmark
    bm_df_in = strategy.benchmark(symbol=symbol,
                                  sd=sd_in,
                                  ed=ed_in,
                                  sv=sv,
                                  commission=commission,
                                  impact=impact)

    # plot
    strategy.plot_all(benchmark=bm_df_in, strategy=portvals_in, buy_date=buy_date_in, sell_date=sell_date_in,
                      title='In-Sample')

    # ----------out of sample period-----------#

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
    strategy.plot_all(benchmark=bm_df_out, strategy=portvals_out, buy_date=buy_date_out, sell_date=sell_date_out,
                      title='Out-of-Sample')

    # ----------Performance Metrics-----------#

    cr_in_port, std_in_port, mean_in_port = strategy.performance_metrics(df_portval=portvals_in)
    cr_in_bm, std_in_bm, mean_in_bm = strategy.performance_metrics(bm_df_in)
    cr_out_port, std_out_port, mean_out_port = strategy.performance_metrics(portvals_out)
    cr_out_bm, std_out_bm, mean_out_bm = strategy.performance_metrics(bm_df_out)

    data_in = [[cr_in_port, std_in_port, mean_in_port],
               [cr_in_bm, std_in_bm, mean_in_bm], ]
    performance_table_in = pd.DataFrame(data_in,
                                        columns=['Cumulative Return', 'STDEV of daily return', 'mean of daily return'],
                                        index=['Portfolio', 'Benchmark'])
    performance_table_in.to_csv('ManulStrategy_InSample_statistics.csv')

    data_out = [[cr_out_port, std_out_port, mean_out_port],
                [cr_out_bm, std_out_bm, mean_out_bm], ]

    performance_table_out = pd.DataFrame(data_out,
                                         columns=['Cumulative Return', 'STDEV of daily return', 'mean of daily return'],
                                         index=['Portfolio', 'Benchmark'])

    performance_table_out.to_csv('ManulStrategy_OutofSample_statistics.csv')
    # suprrot plot:

    strategy.support_plot(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), buy_date=buy_date_in,
                          sell_date=sell_date_in)
