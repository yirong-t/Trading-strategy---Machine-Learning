import pandas as pd
from util import get_data

def compute_portvals(df_trades, start_val=100000, commission=0.00, impact=0.00):



    # Read order file
    orders_df = df_trades
    orders_df = orders_df.sort_index(ascending=True)

    # get all symbols in the order
    symbols = df_trades.columns



    # set date range based on orders.csv file
    start_date = orders_df.index[0].date()
    end_date = orders_df.index[-1].date()

    # get prices for all symbols in that date range
    all_prices = get_data(symbols, pd.date_range(start_date, end_date))
    date_index = all_prices.index


    # create portfoloio data frame
    port_columns = ['portfolio value'] + symbols.tolist() + ['cash']
    portfolio_val = pd.DataFrame(0, index=date_index, columns=port_columns)

    # number of shares
    portfolio_shares = pd.DataFrame(0, index=date_index, columns=symbols.tolist())
    portfolio_val['cash'] = start_val  # initial cash
    # print(portfolio_val.head(2),'\n')
    # print(orders_df[orders_df['JPM']!=0])

    for sym in symbols:
        for i in range(len(orders_df)):

            idx = orders_df.index[i]
            shares = abs(orders_df.loc[idx,sym])
            trading_amount = all_prices.loc[idx, sym] * shares
            impacts = impact * shares * all_prices.loc[idx, sym]

            if orders_df.loc[idx,sym] > 0:
                # long
                # tracking share
                portfolio_shares.loc[idx, sym] += shares
                # tracking cash
                portfolio_val.loc[idx, 'cash'] = portfolio_val.loc[idx, 'cash'] - trading_amount - \
                                                 commission + impacts

            elif orders_df.loc[idx,sym] < 0:
                # short
                # tracking share
                portfolio_shares.loc[idx, sym] -= shares
                # tracking cash
                portfolio_val.loc[idx, 'cash'] = portfolio_val.loc[idx, 'cash'] + trading_amount - \
                                                 commission - impacts
                portfolio_val.loc[idx, sym] = 0

            portfolio_val.loc[idx:, :] = portfolio_val.loc[idx].values
            portfolio_shares.loc[idx:, :] = portfolio_shares.loc[idx].values

    # calculate daily port val for each symbol

    portfolio_val.loc[:, symbols] = all_prices.loc[:, symbols] * portfolio_shares[symbols]
    portfolio_val.iloc[:, 0] = portfolio_val.iloc[:, 1:].sum(axis=1)


    portvals = portfolio_val.iloc[:, [0]]
    rv = pd.DataFrame(index=portvals.index, data=portvals.values)
    # print(portfolio_val)
    return portvals

def statistics(portvals):

    daily_return = (portvals[1:] / portvals[:-1].values) - 1
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        (portvals[-1] - portvals[0]) / portvals[0],
        daily_return.mean(),
        daily_return.std(),
        252 ** 0.5 * (daily_return.mean()) / daily_return.std(),
    ]


    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio




