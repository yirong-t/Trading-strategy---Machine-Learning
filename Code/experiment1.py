from StrategyLearner import *
from ManualStrategy import *
import marketsimcode as msi
import datetime as dt


def benchmark(symbol="JPM",
              sd=dt.datetime(2008, 1, 1),
              ed=dt.datetime(2009, 12, 31),
              sv=100000,
              commission=9.95,
              impact=0.005):

    date_range = pd.date_range(start=sd, end=ed)

    prices_all = get_data([symbol], date_range).bfill().ffill()
    prices = prices_all.iloc[:, 1:]
    date_index = prices.index

    port_columns = ['portfolio value'] + ['JPM'] + ['cash']
    portfolio_val = pd.DataFrame(0, index=date_index, columns=port_columns)

    impacts = impact * 1000 * prices.iloc[0, 0]
    portfolio_val.iloc[0, 2] = sv - prices.iloc[0, 0] * 1000 - impacts - commission
    portfolio_val.iloc[:, 2] = portfolio_val.iloc[0, 2]
    portfolio_val.iloc[:, 1] = prices.iloc[:, 0] * 1000
    portfolio_val.iloc[:, 0] = portfolio_val.sum(axis=1)
    portfolio_val.iloc[-1,2] = portfolio_val.iloc[-1, 0]
    portfolio_val.iloc[-1,1] = 0


    return portfolio_val.iloc[:, 0]


def experiemnt1_plot(symbol='JPM',
                     sd_insample=dt.datetime(2008, 1, 1),
                     ed_insample=dt.datetime(2009, 12, 31),
                     sd_outsample=dt.datetime(2010, 1, 1),
                     ed_outsample=dt.datetime(2011, 12, 31),
                     sv = 100000,
                     impact = 0.005,
                     commission = 9.95):

    # in-sample period
    sd_insample=sd_insample
    ed_insample=ed_insample
    date_range = pd.date_range(sd_insample,ed_insample)


    symbol = symbol


    # benchmark
    protval_benchmark = benchmark(symbol=symbol, sd=sd_insample, ed=ed_insample, sv=sv, commission=commission, impact=impact)

    # Strategy Learner
    learner = StrategyLearner(verbose=False, impact=impact, commission=commission)
    learner.add_evidence(symbol,sd_insample,ed_insample)
    Y_train_strategy = learner.testPolicy(symbol=symbol,sd=sd_insample,ed=ed_insample)
    portval_st_insample = msi.compute_portvals(Y_train_strategy,start_val=sv, commission=commission, impact=impact)


    # Manual Strategy
    strategy = ManualStrategy(verbose=False)
    df_trades = strategy.testPolicy(symbol=symbol,sd=sd_insample,ed=ed_insample)
    portvals_ms_insample = msi.compute_portvals(df_trades,start_val=sv,commission=commission, impact=impact)


    # plot

    protval_benchmark = protval_benchmark/protval_benchmark.iloc[0]
    portval_st_insample = portval_st_insample/portval_st_insample.iloc[0]
    portvals_ms_insample = portvals_ms_insample/portvals_ms_insample.iloc[0]

    plt.figure(figsize=(16,8))
    plt.plot(protval_benchmark,label='benchmark',color='purple')
    plt.plot(portval_st_insample, label='Strategy')
    plt.plot(portvals_ms_insample, label='Manual Strategy',color='red')
    plt.title('In-Sample Performance: Benchmark vs Random Forest vs. Manual Strategy (2008-2009)',fontsize=18)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Normalized Portfolio Value', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig("./images/Experiment1_in_sample_Performance.png")

    # out of sample

    # Strategy
    sd_outsample = sd_outsample
    ed_outsample = ed_outsample

    # benchmark
    protval_benchmark = benchmark(symbol=symbol, sd=sd_outsample, ed=ed_outsample, sv=sv, commission=commission,impact=impact)

    # Strategy
    Y_test_strategy = learner.testPolicy(symbol=symbol, sd=sd_outsample, ed=ed_outsample)
    portval_st_outsample = msi.compute_portvals(Y_test_strategy, start_val=sv, commission=commission, impact=impact)




    # Manual Strategy
    df_trades = strategy.testPolicy(symbol=symbol, sd=sd_outsample, ed=ed_outsample)
    portvals_ms_outsample = msi.compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)



    # plot
    protval_benchmark = protval_benchmark / protval_benchmark.iloc[0]
    portval_st_outsample = portval_st_outsample / portval_st_outsample.iloc[0]
    portvals_ms_outsample = portvals_ms_outsample / portvals_ms_outsample.iloc[0]

    plt.figure(figsize=(16,8))
    plt.plot(protval_benchmark, label='benchmark',color='purple')
    plt.plot(portval_st_outsample, label='Strategy')
    plt.plot(portvals_ms_outsample, label='Manual Strategy',color='red')
    plt.legend()
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Normalized Portfolio Value', fontsize=15)
    plt.grid(True)
    plt.title('Out-of-Sample Performance: Benchmark vs Random Forest vs. Manual Strategy (2010-2011)',fontsize=18)
    plt.savefig("./images/Experiment1_Out_of_Sample_Performance.png")





