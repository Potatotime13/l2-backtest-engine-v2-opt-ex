"""

"""
import pandas as pd
import glob
import os
# import dtale
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
import matplotlib.pyplot as plt
from pandarallel import pandarallel
import seaborn as sns
import yfinance as yf
import json
from scipy.stats import ttest_rel

PATH_ANALYSE = '/Users/markusdiebels/Documents/Schreibtisch/Studium/' \
       'Studium WiInfo/SoSe 2022/AHFT/Analyse'
PATH_TRADES = '/Users/markusdiebels/Documents/Schreibtisch/Studium/' \
              'Studium WiInfo/SoSe 2022/AHFT/Daten/efn2_backtesting/trades'


def get_chart(ticker: str) -> None:
    """Function to plot the historical chart of a stock price.

    :param ticker: Stock ticker
    :return:
    """
    temp_ticker = yf.Ticker(ticker)
    ticker_df = temp_ticker.history(period='1d', start='2021-02-01',
                                    end='2021-02-28')
    temp_df = ticker_df['Close']
    temp_df.plot()
    plt.savefig('results/' + ticker + '.png')
    plt.close()


def create_chart_main() -> None:
    """

    :return:
    """
    tickers = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BMW.DE', 'CON.DE',
               '1COV.DE', 'DDAIF', 'DBK.DE', 'DB1.DE']

    for ticker in tickers:
        get_chart(ticker)

    return None


def add_rel_dif(temp_df: pd.DataFrame, temp_col: str) -> pd.DataFrame:
    temp_df[temp_col + '_dif'] = temp_df[temp_col].pct_change()
    return temp_df


def add_relative_change_csv(folder):
    temp_path = os.path.join(PATH_ANALYSE, folder, '*.csv')
    all_files = glob.glob(temp_path)

    for file in all_files:
        temp_df = pd.read_csv(file)
        temp_df = add_rel_dif(temp_df, 'vwap')
        temp_df = add_rel_dif(temp_df, 'market_vwap')
        temp_df = add_rel_dif(temp_df, 'twap')
        temp_df = add_rel_dif(temp_df, 'market_twap')
        # temp_df.drop(columns=['Unnamed: 0'], inplace=True)
        temp_df.reset_index(drop=True)
        temp_df.to_csv(file)


def create_dataframe_from_folder(folder: str, time_stamp_col='') \
        -> pd.DataFrame:
    """Create a single data frame from all csv-files in one folder

    :param time_stamp_col: Name of column to convert to timestamp
    :param folder: name of the folder
    :return: data frame
    """
    temp_path = os.path.join(PATH_ANALYSE, folder, '*.csv')
    all_files = glob.glob(temp_path)

    temp_df = pd.concat(pd.read_csv(f) for f in all_files)
    if time_stamp_col != '':
        # trades_df
        temp_df[time_stamp_col] = pd.to_datetime(temp_df[time_stamp_col])
        temp_df = temp_df.sort_values(by=['key', 'time'])
        temp_df.drop(columns=['Unnamed: 0'], inplace=True)
    else:
        # mapping_df
        temp_df = temp_df.sort_values(by=['key'])

    temp_df = temp_df.reset_index(drop=True)
    return temp_df


def merge_mean_std_df(temp_df_destination: pd.DataFrame,
                      temp_df_origin: pd.DataFrame, key: str,
                      column_origin: str,  new_name: str) -> pd.DataFrame:
    """Function to merge tradings df and mapping df per parent order
    ans add mean std for each execution

    :param temp_df_destination: df to merge and group to
    :param temp_df_origin: df with data to group an calculate
    :param key: column name to merge
    :param column_origin: name of the column to calculate
    :param new_name: name of the result column in destination df
    :return: df with results
    """
    # Add mean
    temp_df_destination = temp_df_destination.merge(
        temp_df_origin.groupby([key])[column_origin].mean(), left_on=key,
        right_on=key, validate='1:1')
    temp_df_destination.rename(columns={column_origin: new_name + '_mean'},
                               inplace=True)

    # Add std
    temp_df_destination = temp_df_destination.merge(
        temp_df_origin.groupby([key])[column_origin].std(),
        left_on=key, right_on=key, validate='1:1')
    temp_df_destination.rename(columns={column_origin: new_name + '_std'},
                               inplace=True)
    return temp_df_destination


def add_volume(stock: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Function to add the in a time window traded volume to the mapping df.

    :param stock: stock name
    :param start: timestamp start
    :param end: timestamp end
    :return: traded volume in time window
    """
    temp_stock_name = stock
    if temp_stock_name == 'test':
        return 0
    temp_date = str(start.date()).replace('-', '')
    temp_file_name = 'TRADES_' + temp_stock_name + '_DE_' \
                     + temp_date + '_' + temp_date + '.json'
    temp_file = os.path.join(PATH_TRADES, temp_file_name)

    with open(temp_file) as json_file:
        temp_json = json.load(json_file)

    temp_sum = 0

    if len(temp_json) > 0:
        data = flatten_data(temp_json)
        data_df = pd.DataFrame(data)
        data_df['TIMESTAMP_UTC'] = pd.to_datetime(data_df['TIMESTAMP_UTC'])

        temp_sum += data_df.loc[(data_df['TIMESTAMP_UTC'] >= start) &
                                (data_df['TIMESTAMP_UTC'] <= end),
                                'Volume'].sum()

    return temp_sum


def flatten_data(data: dict) -> list:
    """

    :param data: Dictionary from json file
    :return: list with values to create a pandas data frame
    """
    data_flat = []
    for d in data:
        if len(d['Price']) > 1:
            for i in range(len(d['Price'])):
                data_flat.append({'TIMESTAMP_UTC': d['TIMESTAMP_UTC'],
                                  'Price': d['Price'][i],
                                  'Volume': d['Volume'][i]})
        else:
            data_flat.append({'TIMESTAMP_UTC': d['TIMESTAMP_UTC'],
                              'Price': d['Price'][0],
                              'Volume': d['Volume'][0]})
    return data_flat


def get_trades(path):
    """Get trades from json file to data frame

    :param path: json file path
    :return:
    """
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    if len(data) > 0:
        data = flatten_data(data)
        data_df = pd.DataFrame(data)
        data_df['TIMESTAMP_UTC'] = pd.to_datetime(data_df['TIMESTAMP_UTC'])
        data_df.set_index('TIMESTAMP_UTC', inplace=True)
        return data_df
    else:
        return None


def add_t_values(key: str, dif: str, temp_df: pd.DataFrame) -> tuple:
    """

    :param key: key for mapping and trades df
    :param dif: dif_vwap_rel or dif_twap_rel
    :param temp_df: trades_df
    :return: t-stat value und probability
    """
    vgl = [0] * len(temp_df.loc[temp_df['key'] == key][dif])

    stat, p = ttest_rel(vgl, temp_df.loc[temp_df['key'] == key][dif])
    return stat, p


def add_direction_label(key: int, time_window: int,
                        temp_trades_df: pd.DataFrame) -> int:
    """Function to label the change with in a time intervall
    of a midpoint in rising, falling or constant price.

    :param key: key in mapping_df and trades_df
    :param time_window: 3h or 6h
    :param temp_trades_df: df with trade values
    :return: int as label for intervall interpretation
    """
    if key == 0:
        return 0
    temp_df = temp_trades_df.loc[temp_trades_df['key'] == key]['midpoint']
    temp_df_len = len(temp_df)
    temp_start_midpoint = temp_df.iloc[0]
    temp_df = temp_df - temp_start_midpoint
    temp_label = temp_df.sum() / (temp_df_len * temp_start_midpoint)

    if time_window == 3:
        if temp_label <= -0.002:
            temp_label = -1
        elif temp_label >= 0.002:
            temp_label = 1
        else:
            temp_label = 0
    else:  # 6h
        if temp_label <= -0.0025:
            temp_label = -1
        elif temp_label >= 0.0025:
            temp_label = 1
        else:
            temp_label = 0

    return temp_label


def replace_key() -> None:
    """Function to create a unique key over all result csv files.

    :return: None
    """
    max_key = 0
    temp_path = os.path.join(PATH_ANALYSE, 'Input_Data')
    for folder in os.listdir(temp_path):
        if os.path.isdir(os.path.join(temp_path, folder)):
            temp_path_trades = os.path.join(temp_path, folder, '*.csv')
            all_trades = glob.glob(temp_path_trades)
            temp_path_mappings = os.path.join(temp_path, folder,
                                              'mappings', '*.csv')
            all_mappings = glob.glob(temp_path_mappings)
            all_trades.sort()
            all_mappings.sort()
            for i in range(len(all_mappings)):
                print(str(max_key))
                print(str(i))
                print(all_trades[i])
                print(all_mappings[i])

                temp_name = all_trades[i]
                temp_df = pd.read_csv(temp_name)
                os.remove(temp_name)
                temp_name = temp_name.replace('.csv', '_' + str(max_key)
                                              + '.csv')
                temp_df['key'] = max_key
                temp_df.to_csv(temp_name, index=False)

                temp_name = all_mappings[i]
                temp_df = pd.read_csv(temp_name)
                os.remove(temp_name)
                temp_name = temp_name.replace('.csv', '_' + str(max_key)
                                              + '.csv')
                temp_df['key'] = max_key
                temp_df.to_csv(temp_name, index=False)
                max_key += 1


def regression(temp_df: pd.DataFrame, side: str, dependent: str, factor: int,
               title: str, reg_name: str) -> None:
    """Function to create regression in different versions.

    :param temp_df: mapping_df
    :param side: buy or sell
    :param dependent: dependent variable for regression
    :param factor: to calculate as bps
    :param title: result tabular title
    :param reg_name: tex file name
    :return: None
    """
    temp_df_fall = temp_df.loc[(temp_df['side'] == side)
                               & (temp_df['direction_label'] == -1)]
    result_fall = sm.OLS(endog=temp_df_fall[dependent] * factor,
                         exog=temp_df_fall[['agent', 'time_dummy',
                                            'rel_vol', 'midpoint_std']],
                         missing='drop').fit()
    temp_df_const = temp_df.loc[(temp_df['side'] == side)
                                & (temp_df['direction_label'] == 0)]
    result_const = sm.OLS(endog=temp_df_const[dependent] * factor,
                          exog=temp_df_const[['agent', 'time_dummy',
                                              'rel_vol', 'midpoint_std']],
                          missing='drop').fit()
    temp_df_rise = temp_df.loc[(temp_df['side'] == side)
                               & (temp_df['direction_label'] == 1)]
    result_rise = sm.OLS(endog=temp_df_rise[dependent] * factor,
                         exog=temp_df_rise[['agent', 'time_dummy',
                                            'rel_vol', 'midpoint_std']],
                         missing='drop').fit()
    stargazer = Stargazer([result_fall, result_const, result_rise])
    stargazer.rename_covariates({'agent': 'Agent', 'midpoint_std': 'Std',
                                 'rel_vol': 'Volumen',
                                 'time_dummy': 'Intervall'})
    stargazer.title(title)
    stargazer.custom_columns(['Fallend', 'Konstant', 'Steigend'], [1, 1, 1])
    file_name = os.path.join('results', reg_name + '.tex')
    tex_file = open(file_name, 'w')
    tex_file.write(stargazer.render_latex())
    tex_file.close()

    return None


def regression_trades(temp_df: pd.DataFrame, side: str) -> None:
    temp_df_fall = temp_df.loc[(temp_df['side'] == side)
                               & (temp_df['direction_label'] == -1)]
    result_fall = sm.OLS(endog=temp_df_fall['vwap'],
                         exog=temp_df_fall[['market_vwap', 'agent']],
                         missing='drop').fit()
    temp_df_const = temp_df.loc[(temp_df['side'] == side)
                                & (temp_df['direction_label'] == 0)]
    result_const = sm.OLS(endog=temp_df_const['vwap'],
                          exog=temp_df_const[['market_vwap', 'agent']],
                          missing='drop').fit()
    temp_df_rise = temp_df.loc[(temp_df['side'] == side)
                               & (temp_df['direction_label'] == 1)]
    result_rise = sm.OLS(endog=temp_df_rise['vwap'],
                         exog=temp_df_rise[['market_vwap', 'agent']],
                         missing='drop').fit()
    stargazer = Stargazer([result_fall, result_const, result_rise])
    stargazer.rename_covariates({'market_vwap': 'Markt VWAP'})
    stargazer.title('Trades Regression VWAP ' + side)
    stargazer.custom_columns(['Fallend', 'Konstant', 'Steigend'], [1, 1, 1])
    file_name = os.path.join('results', 'Trades_VWAP_' + side + '.tex')
    tex_file = open(file_name, 'w')
    tex_file.write(stargazer.render_latex())
    tex_file.close()


def stock_statistics(temp_df: pd.DataFrame, stock_name: str) -> None:
    first_price = temp_df.iloc[0, 0]
    last_price = temp_df.iloc[-1, 3]
    high_price = temp_df["High"].max()
    low_price = temp_df["Low"].min()
    average_price = (temp_df['Open'].mean() + temp_df['Close'].mean()) / 2
    print(f'Eröffnungspreis {stock_name} {start_date}: {first_price}')
    print(f'Endpreis {stock_name} {end_date}: {last_price}')
    print(f'Preisänderung {stock_name} von '
          f'{(last_price/first_price - 1) * 100} %')
    print(f'Durchschnittlicher Pries {stock_name} von {average_price}')
    print(f'Maximaler Preis {stock_name} von {high_price}')
    print(f'Max Preis relative Abweichung {stock_name} vom Durchschnitt '
          f'{(high_price/average_price - 1) * 100} %')
    print(f'Minimaler Preis {stock_name} von {low_price}')
    print(f'Min Preis relative Abweichung {stock_name} vom Durchschnitt '
          f'{(low_price / average_price - 1) * 100} %')
    print('\n')


def parent_order_statistics(temp_df: pd.DataFrame, stock_name: str) -> None:
    poc_volume_stock = temp_df.loc[temp_df['stock'] == stock_name,
                                   'abs_vol'].sum()
    poc_volume_episode = temp_df.loc[temp_df['stock'] == stock_name,
                                     'volume'].sum()
    poc_agent0_stock = len(temp_df.loc[
                                (temp_df['agent'] == 0) &
                                (temp_df['stock'] == stock_name)
                                ].index)
    poc_agent0_stock_volume = temp_df.loc[
                               (temp_df['agent'] == 0) &
                               (temp_df['stock'] == stock_name),
                               'abs_vol'
                               ].sum()
    poc_agent0_episode_volume = temp_df.loc[(temp_df['agent'] == 0) &
                                            (temp_df['stock'] == stock_name),
                                            'volume'].sum()
    poc_agent0_in_time = temp_df.loc[(temp_df['agent'] == 0) &
                                     (temp_df['stock'] == stock_name),
                                     'in_time'].sum()
    poc_agent0_stock_buy = len(temp_df.loc[
                                   (temp_df['agent'] == 0) &
                                   (temp_df['stock'] == stock_name) &
                                   (temp_df['side'] == 'buy')
                                   ].index)
    poc_agent0_stock_sell = len(mapping_df.loc[
                                    (mapping_df['agent'] == 0) &
                                    (mapping_df['stock'] == stock_name) &
                                    (mapping_df['side'] == 'sell')
                                    ].index)

    poc_agent1_episode_volume = temp_df.loc[(temp_df['agent'] == 1) &
                                            (temp_df['stock'] == stock_name),
                                            'volume'].sum()
    poc_agent1_stock = len(temp_df.loc[
                                (temp_df['agent'] == 1) &
                                (temp_df['stock'] == stock_name)
                                ].index)
    poc_agent1_stock_volume = temp_df.loc[(temp_df['agent'] == 0) &
                                          (temp_df['stock'] == stock_name),
                                          'abs_vol'].sum()
    poc_agent1_in_time = temp_df.loc[(temp_df['agent'] == 1) &
                                     (temp_df['stock'] == stock_name),
                                     'in_time'].sum()
    poc_agent1_stock_buy = len(temp_df.loc[
                                   (temp_df['agent'] == 1) &
                                   (temp_df['stock'] == stock_name) &
                                   (temp_df['side'] == 'buy')
                                   ].index)
    poc_agent1_stock_sell = len(mapping_df.loc[
                                    (mapping_df['agent'] == 1) &
                                    (mapping_df['stock'] == stock_name) &
                                    (mapping_df['side'] == 'sell')
                                    ].index)
    print(f'Insgesamt wurde für {stock_name} ein Volumen von '
          f'{poc_volume_stock} gehandelt')
    print(f'Innerhalb der betrachteten Episoden wurden ein Volumen von '
          f'{poc_volume_episode} gehandelt. '
          f'Anteil Agent: {poc_volume_stock/poc_volume_episode * 100} %')

    print(f'Insgesamt für Agent 0 und {stock_name} {poc_agent0_stock} '
          f'Orders mit einem Volumen von {poc_agent0_stock_volume} ausgeführt')
    print(f'Innerhalb der betrachteten Episoden wurden ein Volumen von '
          f'{poc_agent0_episode_volume} gehandelt. '
          f'Anteil Agent: '
          f'{poc_agent0_stock_volume / poc_agent0_episode_volume * 100} %')
    print(f'Rechtzeitig ausgeführt: {poc_agent0_in_time}')
    print(f'Nicht Rechtzeitig ausgeführt: '
          f'{poc_agent0_in_time - poc_agent0_stock}')
    print(f'Davon waren {poc_agent0_stock_buy} buy orders')
    print(f'und {poc_agent0_stock_sell} sell orders')

    print(f'Insgesamt für Agent 1 und {stock_name} {poc_agent1_stock} '
          f'Orders mit einem Volumen von {poc_agent1_stock_volume} ausgeführt')
    print(f'Innerhalb der betrachteten Episoden wurden ein Volumen von '
          f'{poc_agent1_episode_volume} gehandelt.'
          f'Anteil Agent: '
          f'{poc_agent1_stock_volume / poc_agent1_episode_volume * 100} %')
    print(f'Rechtzeitig ausgeführt: {poc_agent1_in_time}')
    print(f'Nicht Rechtzeitig ausgeführt: '
          f'{poc_agent1_in_time - poc_agent1_stock}')
    print(f'Davon waren {poc_agent1_stock_buy} buy orders')
    print(f'und {poc_agent1_stock_sell} sell orders')
    print('\n')


def market_statistics(temp_df: pd.DataFrame, stock_name: str,
                      agent: int) -> None:
    ms_dif01 = len(temp_df.loc[(temp_df['agent'] == agent) &
                               (temp_df['stock'] == stock_name) &
                               (temp_df['p_vwap'] < 0.1)
                               ].index)
    ms_dif005 = len(temp_df.loc[(temp_df['agent'] == agent) &
                                (temp_df['stock'] == stock_name) &
                                (temp_df['p_vwap'] < 0.05)
                                ].index)
    ms_dif001 = len(temp_df.loc[(temp_df['agent'] == agent) &
                                (temp_df['stock'] == stock_name) &
                                (temp_df['p_vwap'] < 0.01)
                                ].index)
    ms_buy = len(temp_df.loc[(temp_df['agent'] == agent) &
                             (temp_df['stock'] == stock_name) &
                             (temp_df['side'] == 'buy')
                             ].index)
    ms_better_buy = len(temp_df.loc[(temp_df['agent'] == agent) &
                                    (temp_df['stock'] == stock_name) &
                                    (temp_df['side'] == 'buy') &
                                    (temp_df['dif_vwap_rel_mean'] < 0)
                                    ].index)
    ms_better_buy_rise = len(temp_df.loc[(temp_df['agent'] == agent) &
                                         (temp_df['stock'] == stock_name) &
                                         (temp_df['side'] == 'buy') &
                                         (temp_df['dif_vwap_rel_mean'] < 0) &
                                         (temp_df['direction_label'] == 1)
                                         ].index)
    ms_sell = len(temp_df.loc[(temp_df['agent'] == agent) &
                              (temp_df['stock'] == stock_name) &
                              (temp_df['side'] == 'sell')
                              ].index)
    ms_better_sell = len(temp_df.loc[(temp_df['agent'] == agent) &
                                     (temp_df['stock'] == stock_name) &
                                     (temp_df['side'] == 'sell') &
                                     (temp_df['dif_vwap_rel_mean'] > 0)
                                     ].index)
    ms_better_sell_fall = len(temp_df.loc[(temp_df['agent'] == agent) &
                                          (temp_df['stock'] == stock_name) &
                                          (temp_df['side'] == 'sell') &
                                          (temp_df['dif_vwap_rel_mean'] > 0) &
                                          (temp_df['direction_label'] == -1)
                                          ].index)
    ms_avg_dif_buy = temp_df.loc[(temp_df['agent'] == agent) &
                                 (temp_df['stock'] == stock_name) &
                                 (temp_df['side'] == 'buy'),
                                 'dif_vwap_abs_mean'
                                 ].mean()
    ms_avg_dif_sell = temp_df.loc[(temp_df['agent'] == agent) &
                                  (temp_df['stock'] == stock_name) &
                                  (temp_df['side'] == 'sell'),
                                  'dif_vwap_abs_mean'
                                  ].mean()
    print(f'Aktie: {stock_name}, Agent {agent}')
    print(f'Bei {ms_dif01} Orders hat sich ein zum 10% Konfidenzintervall '
          f'anderes Ergebnis als am Markt ergeben')
    print(f'Bei {ms_dif005} Orders hat sich ein zum 5% Konfidenzintervall '
          f'anderes Ergebnis als am Markt ergeben')
    print(f'Bei {ms_dif001} Orders hat sich ein zum 1% Konfidenzintervall '
          f'anderes Ergebnis als am Markt ergeben')
    print(f'Besser gekauft als der Markt (delta < 0): '
          f'{ms_better_buy} von {ms_buy}')
    print(f'davon bei steigenden Kursen: {ms_better_buy_rise}')
    print(f'Durchschnittliche absolute buy VWAP Abweichung '
          f'zum Markt: {ms_avg_dif_buy}')
    print(f'Besser verkauft als der Markt (delta > 0): '
          f'{ms_better_sell} von {ms_sell}')
    print(f'davon bei fallenden Kursen: {ms_better_sell_fall}')
    print(f'Durchschnittliche absolute sell VWAP Abweichung '
          f'zum Markt: {ms_avg_dif_sell}')

    print('\n')


def agent_statistics(temp_df: pd.DataFrame, agent: int, side: str) -> None:
    avg_dif = temp_df.loc[(temp_df['agent'] == agent) &
                          (temp_df['side'] == side),
                          'dif_vwap_abs_mean'].mean()
    print(f'{side} Order durchschnittliche '
          f'Abweichung Agent {agent}: {avg_dif}')


# replace_key()
# Add relative change values to single trade.csv
# add_relative_change_csv('save_games')

# Create df
mapping_df = create_dataframe_from_folder('mappings')
trades_df = create_dataframe_from_folder('save_games', 'time')


# Add tick size
mapping_df['tick_size'] = 0.05  # Allianz and Continental
mapping_df.loc[mapping_df.stock == 'Adidas', ['tick_size']] = 0.1

# Add start time and end time of order execution
mapping_df = mapping_df.merge(trades_df.groupby(['key'])['time'].min(),
                              left_on='key', right_on='key', validate='1:1')
mapping_df.rename(columns={'time': 'start'}, inplace=True)
mapping_df = mapping_df.merge(trades_df.groupby(['key'])['time'].max(),
                              left_on='key', right_on='key', validate='1:1')
mapping_df.rename(columns={'time': 'end'}, inplace=True)
mapping_df['duration'] = (mapping_df['end']
                          - mapping_df['start']).astype('timedelta64[h]')
mapping_df['in_time'] = mapping_df['time_window'] > mapping_df['duration']

# Add traded volume of time window
# might need to be executed via terminal because of pandarallel
pandarallel.initialize()
mapping_df['volume'] = mapping_df.parallel_apply(
    lambda x: add_volume(x['stock'], x['start'], x['end']), axis=1)

# Add relative traded volume
mapping_df['rel_vol'] = mapping_df['abs_vol'] / mapping_df['volume']

# Add rel and abs difference vwap and twap
trades_df['dif_vwap_rel'] = (trades_df['vwap'] / trades_df['market_vwap']) - 1
trades_df['dif_twap_rel'] = (trades_df['twap'] / trades_df['market_twap']) - 1

trades_df['dif_vwap_abs'] = trades_df['vwap'] - trades_df['market_vwap']
trades_df['dif_twap_abs'] = trades_df['twap'] - trades_df['market_twap']

# Add mean rel abs for vwap twap
mapping_df = merge_mean_std_df(mapping_df, trades_df, 'key',
                               'dif_vwap_rel', 'dif_vwap_rel')
mapping_df = merge_mean_std_df(mapping_df, trades_df, 'key',
                               'dif_vwap_abs', 'dif_vwap_abs')
mapping_df = merge_mean_std_df(mapping_df, trades_df, 'key',
                               'dif_twap_rel', 'dif_twap_rel')
mapping_df = merge_mean_std_df(mapping_df, trades_df, 'key',
                               'dif_twap_abs', 'dif_twap_abs')
# Add midpoint std
mapping_df = merge_mean_std_df(mapping_df, trades_df, 'key',
                               'midpoint', 'midpoint')

# T Test for significant difference from vwap/twap to 0
# vwap

temp_stat_df = mapping_df.parallel_apply(
    lambda x: add_t_values(x['key'], 'dif_vwap_rel', trades_df),
    axis='columns', result_type='expand')
mapping_df = pd.concat([mapping_df, temp_stat_df], axis='columns')
mapping_df.rename({0: 'stat_vwap', 1: 'p_vwap'}, axis='columns', inplace=True)
# twap
temp_stat_df = mapping_df.parallel_apply(
    lambda x: add_t_values(x['key'], 'dif_twap_rel', trades_df),
    axis='columns', result_type='expand')
mapping_df = pd.concat([mapping_df, temp_stat_df], axis='columns')
mapping_df.rename({0: 'stat_twap', 1: 'p_twap'}, axis='columns', inplace=True)

# Add label for intervalls with falling (-1), constant (0), rising (1) prices
mapping_df['direction_label'] = mapping_df.parallel_apply(
    lambda x: add_direction_label(x['key'], x['time_window'],
                                  trades_df), axis=1)

mapping_df.to_csv('mappings.csv')
trades_df.to_csv('trades.csv')


# Write statistic
start_date = str(mapping_df['start'].min()).split()[0]
end_date = str(mapping_df['end'].max()).split()[0]
print(f'Erste Order am: {start_date}')
print(f'Letzte Order am {end_date}')

# Get data for stock chart
adidas_ticker = yf.Ticker('ADS.DE')
adidas_df = adidas_ticker.history(period='1d', start=start_date,
                                  end=end_date)
allianz_ticker = yf.Ticker('ALV.DE')
allianz_df = allianz_ticker.history(period='1d', start=start_date,
                                    end=end_date)
continental_ticker = yf.Ticker('CON.DE')
continental_df = continental_ticker.history(period='1d', start=start_date,
                                            end=end_date)
# Plot stock charts
fig, axs = plt.subplots(3, figsize=(10, 10))
fig.suptitle('Aktien Preis Entwicklung Februar 2021')
axs[0].plot(adidas_df.index, adidas_df['Close'])
axs[0].set_title('Adidas')
axs[1].plot(allianz_df.index, allianz_df['Close'], 'tab:orange')
axs[1].set_title('Allianz')
axs[2].plot(continental_df.index, continental_df['Close'], 'tab:green')
axs[2].set_title('Continental')


for ax in axs.flat:
    ax.set(xlabel='Datum', ylabel='Schlusspreis in EUR')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.savefig('results/stocks.png')
plt.close()

# Print stock price statistics
stock_statistics(adidas_df, 'Adidas')
stock_statistics(allianz_df, 'Allianz')
stock_statistics(continental_df, 'Continental')

# Print agent statistics
parent_order_count = len(mapping_df.index)
print(f'Insgesamt wurden {parent_order_count} parent orders ausgeführt')
parent_order_in_time = mapping_df['in_time'].sum()
print(f'Davon wurden {parent_order_in_time} Orders innerhalb der '
      f'vorgegebenen Zeit ausgeführt.')
print(f'{parent_order_count - parent_order_in_time} Orders wurden nicht'
      f'rechtzeitig ausgeführt, abweichung < 1 Minute.')

parent_order_statistics(mapping_df, 'Adidas')
parent_order_statistics(mapping_df, 'Allianz')
parent_order_statistics(mapping_df, 'Continental')

child_order_count = len(trades_df.index)

merged_trades_df = trades_df.merge(mapping_df, left_on='key',
                                   right_on='key', validate='m:1')

child_order_count_agent0 = len(merged_trades_df.loc[
                                   (merged_trades_df['agent'] == 0)].index)
child_order_count_agent1 = len(merged_trades_df.loc[
                                   (merged_trades_df['agent'] == 1)].index)
print(f'Insgesamt wurden dazu {child_order_count} child orders ausgeführt')
print(f'Davon entfalle {child_order_count_agent0} auf Agent 0')
print(f'und {child_order_count_agent1} auf Agent 1')
total_episode_time = mapping_df['time_window'].sum()
total_episode_time_agent0 = mapping_df.loc[(mapping_df['agent'] == 0),
                                           'time_window'].sum()
total_episode_time_agent1 = mapping_df.loc[(mapping_df['agent'] == 1),
                                           'time_window'].sum()
print(f'Bei einer gesamten Ausführungszeit von {total_episode_time} '
      f'Stunden, entspricht das einer durchschnittlichen Ausführung von '
      f'{child_order_count / total_episode_time} Orders pro Stunde')
print(f'Für Agent 0 entspricht das '
      f'{child_order_count_agent0 / total_episode_time_agent0} '
      f'Orders pro Stunde bei {total_episode_time_agent0} Stunden')
print(f'Für Agent 1 entspricht das '
      f'{child_order_count_agent1 / total_episode_time_agent1} '
      f'Orders pro Stunde bei {total_episode_time_agent1}')


# Compare Agent orders with market
market_statistics(mapping_df, 'Adidas', 0)
market_statistics(mapping_df, 'Adidas', 1)
market_statistics(mapping_df, 'Allianz', 0)
market_statistics(mapping_df, 'Allianz', 1)
market_statistics(mapping_df, 'Continental', 0)
market_statistics(mapping_df, 'Continental', 1)

agent_statistics(mapping_df, 0, 'buy')
agent_statistics(mapping_df, 0, 'sell')
agent_statistics(mapping_df, 1, 'buy')
agent_statistics(mapping_df, 1, 'sell')

mapping_df['time_dummy'] = 0
mapping_df.loc[mapping_df['time_window'] == 6, 'time_dummy'] = 1

# Regression vwap buy
regression(mapping_df, 'buy', 'dif_vwap_abs_mean', 10000,
           'VWAP Buy abs', 'buy_vwap_abs')
regression(mapping_df, 'sell', 'dif_vwap_abs_mean', 10000,
           'VWAP Sell abs', 'sell_vwap_abs')

regression(mapping_df, 'buy', 'dif_twap_abs_mean', 10000,
           'TWAP Buy abs', 'buy_twap_abs')
regression(mapping_df, 'sell', 'dif_twap_abs_mean', 10000,
           'TWAP Sell abs', 'sell_twap_abs')

trades_df = trades_df.merge(mapping_df, left_on='key',
                            right_on='key', validate='m:1')

regression_trades(trades_df, 'buy')
regression_trades(trades_df, 'sell')

# Boxplot with VWAP and TWAP per agent
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
fig.suptitle('Durchschnittliche relative Abweichung zum Markt')
sns.boxplot(ax=axs[0, 0], x=mapping_df['agent'],
            y=mapping_df.loc[mapping_df['side'] == 'buy',
                             'dif_vwap_rel_mean']*100)
axs[0, 0].set_title('VWAP Buy')
sns.boxplot(ax=axs[0, 1], x=mapping_df['agent'],
            y=mapping_df.loc[mapping_df['side'] == 'sell',
                             'dif_vwap_rel_mean']*100)
axs[0, 1].set_title('VWAP Sell')
sns.boxplot(ax=axs[1, 0], x=mapping_df['agent'],
            y=mapping_df.loc[mapping_df['side'] == 'buy',
                             'dif_twap_rel_mean']*100)
axs[1, 0].set_title('TWAP Buy')
sns.boxplot(ax=axs[1, 1], x=mapping_df['agent'],
            y=mapping_df.loc[mapping_df['side'] == 'sell',
                             'dif_twap_rel_mean']*100)
axs[1, 1].set_title('TWAP Sell')

for ax in axs.flat:
    ax.set(xlabel='Agent', ylabel='Abweichung in ')

for ax in axs.flat:
    ax.label_outer()

plt.savefig('results/boxplot.png')
plt.close()
