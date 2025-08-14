
from StrategyLearner import *
from ManualStrategy import *
import marketsimcode as msi
import datetime as dt


def performance_metrics(df_portval=None):

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

def experiment2(symbol='JPM',
                impact_1 = 0,
                impact_2 = 0.05,
                impact_3 = 0.1,
                sd = dt.datetime(2008, 1, 1),
                ed = dt.datetime(2009, 12, 31),
                sv=100000):




    # in-sample period
    sd_insample=sd
    ed_insample=ed



    # impact 1
    learner1 = StrategyLearner(verbose=False, impact=impact_1, commission=0)
    learner1.add_evidence(symbol, sd_insample, ed_insample)
    Y_train_strategy1 = learner1.testPolicy(symbol=symbol, sd=sd_insample, ed=ed_insample)
    portval_st_insample1 = msi.compute_portvals(Y_train_strategy1, start_val=sv, commission=0, impact=impact_1)
    cr_insampe1, std_nsampe1, mean_insampe1 = performance_metrics(portval_st_insample1)

    # impact 2
    learner2 = StrategyLearner(verbose=False, impact=impact_2, commission=0)
    learner2.add_evidence(symbol, sd_insample, ed_insample)
    Y_train_strategy2 = learner2.testPolicy(symbol=symbol, sd=sd_insample, ed=ed_insample)
    portval_st_insample2 = msi.compute_portvals(Y_train_strategy2, start_val=sv, commission=0, impact=impact_2)
    cr_insample2,std_nsampe2, mean_insampe2 = performance_metrics(portval_st_insample2)

    # impact 3
    learner3 = StrategyLearner(verbose=False, impact=impact_3, commission=0)
    learner3.add_evidence(symbol, sd_insample, ed_insample)
    Y_train_strategy3 = learner3.testPolicy(symbol=symbol, sd=sd_insample, ed=ed_insample)
    portval_st_insample3 = msi.compute_portvals(Y_train_strategy3, start_val=sv, commission=0, impact=impact_3)
    cr_insample3, std_nsampe3, mean_insampe3 = performance_metrics(portval_st_insample3)

    # print(portval_st_insample1)

    portval_st_insample1 = portval_st_insample1/portval_st_insample1.iloc[0]
    portval_st_insample2 = portval_st_insample2/portval_st_insample2.iloc[0]
    portval_st_insample3 = portval_st_insample3/portval_st_insample3.iloc[0]

    plt.figure(figsize=(16, 8))
    plt.plot(portval_st_insample1, label=f'impact={impact_1}', color='purple')
    plt.plot(portval_st_insample2, label=f'impact={impact_2}', color='blue')
    plt.plot(portval_st_insample3, label=f'impact={impact_3}', color='red')
    plt.title("Impact",fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Normalized Portfolio Value', fontsize=15)
    plt.grid(True)
    plt.legend([f'impact_1={impact_1}', f'impact_2={impact_2}', f'impact_3={impact_3}'], fontsize=15)
    plt.savefig("./images/Experiment2.png")

    data = [[cr_insampe1, std_nsampe1, mean_insampe1],
            [cr_insample2, std_nsampe2, mean_insampe2],
            [cr_insample3, std_nsampe3, mean_insampe3],]

    statistics_df=pd.DataFrame(data, columns=['Cumulative Return', 'STDEV of daily return', 'mean of daily return'],index=[f'Impact1={impact_1}',f'Impact2={impact_2}',f'Impact3={impact_3}'])

    statistics_df.to_csv("Experiment2_statistics.csv")

