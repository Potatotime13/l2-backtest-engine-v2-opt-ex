# create stock list
# iterate stocks
# iterate days
# read json to dict
# create day summary
# store day summayr in list
# save combined summary for stock

# importing the module
import json
import numpy as np
import pandas as pd
import os
import unicodedata
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_file_names(stock, dir_list):
    tmp_dir_list = []
    for file_name in dir_list:
        if stock in file_name:
            tmp_dir_list.append(file_name)
    return tmp_dir_list

def flatten_data(data:dict):
    data_flat = []
    for d in data:
        if len(d['Price'])>1:
            for i in range(len(d['Price'])):
                data_flat.append({'TIMESTAMP_UTC':d['TIMESTAMP_UTC'], 'Price':d['Price'][i], 'Volume':d['Volume'][i]})
        else:
            data_flat.append({'TIMESTAMP_UTC':d['TIMESTAMP_UTC'], 'Price':d['Price'][0], 'Volume':d['Volume'][0]})
    return data_flat

def get_raw_book(path):
    data_tmp = pd.read_csv(path, compression='gzip')
    data_tmp['TIMESTAMP_UTC'] = pd.to_datetime(data_tmp['TIMESTAMP_UTC'])
    data_tmp.set_index('TIMESTAMP_UTC', inplace=True)
    return data_tmp

def get_midpoints(path):
    data_tmp = pd.read_csv(path, compression='gzip')
    data_tmp['TIMESTAMP_UTC'] = pd.to_datetime(data_tmp['TIMESTAMP_UTC'])
    data_tmp.set_index('TIMESTAMP_UTC', inplace=True)
    out =  data_tmp.loc[:,['L1-AskPrice','L1-BidPrice']].mean(axis=1)
    out.dropna(inplace=True)
    return out

def get_trades(path):
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

def get_dirlist(book:bool):
    path_trades = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/trades/'
    path_book = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/book/'
    dir_list_t = [unicodedata.normalize('NFC', f) for f in os.listdir(path_trades)]
    dir_list_b = [unicodedata.normalize('NFC', f) for f in os.listdir(path_book)]

    if book:
        return dir_list_b, path_book
    else:
        return dir_list_t, path_trades

def create_volume_distribution(stock_list, dir_list_t, dir_list_b):
    vol_dist = {}
    for stock in tqdm(stock_list):
        files_t = get_file_names(stock,dir_list_t)[:7]
        files_b = get_file_names(stock,dir_list_b)[:7]
        avg_dist = None
        count = 0
        for file_t, file_b in zip(files_t,files_b):
            path_t = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/trades/'+ file_t
            path_b = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/book/'+ file_b
            mid = get_midpoints(path_b)
            trade = get_trades(path_t)
            if not trade is None:
                edges = trade.copy()
                for i in range(trade.shape[0]):
                    ind = trade.index[i]
                    timestamp = mid.index[mid.index.get_loc(ind)-1]
                    edges.loc[ind,'Price'] = np.abs(trade.loc[ind,'Price']-mid[timestamp])
                edges = edges.round({'Price':2,'Volume':0})
                dist = edges.groupby('Price').sum()
                if avg_dist is None:
                    avg_dist = dist
                else:
                    avg_dist = avg_dist.join(dist, rsuffix=str(count))
                count += 1
        vol_dist[stock] = avg_dist.mean(axis=1)
        vol_dist[stock].index = vol_dist[stock].index.set_names('edge')
    return vol_dist

def create_volume_summary(stock_list, dir_list):
    volume_summary = {}
    for stock in tqdm(stock_list):
        files = get_file_names(stock,dir_list)[:7]
        hour_avg = None
        for file in files:
            with open(u'C:/Users/Lucas/Downloads/test_trade_download/trades/'+file, 'r') as json_file:
                data = json.load(json_file)
            if len(data) > 0:
                data_df = pd.DataFrame(data)
                data_df['TIMESTAMP_UTC'] = pd.to_datetime(data_df['TIMESTAMP_UTC'])
                data_df['hours'] = data_df['TIMESTAMP_UTC'].dt.hour
                data_df['Volume_agr'] = data_df.apply(lambda row : sum(row['Volume']), axis = 1)
                summary = data_df[['hours','Volume_agr']].groupby(['hours']).sum()
                if hour_avg is None:
                    hour_avg = summary['Volume_agr'].to_numpy()
                else:
                    hour_avg += summary['Volume_agr'].to_numpy()
        volume_summary[stock] = (hour_avg/len(files)).tolist()
    return volume_summary

def create_vola_summary(stock_list, dir_list):
    vola_summary = {}
    for stock in tqdm(stock_list):
        files = get_file_names(stock,dir_list)
        hour_avg = None
        for file in files:
            with open(u'C:/Users/Lucas/Downloads/test_trade_download/trades/'+file, 'r') as json_file:
                data = json.load(json_file)
            if len(data) > 0:
                data_flat = flatten_data(data)
                data_df = pd.DataFrame(data_flat)
                data_df['Multiply'] = data_df['Price'] * data_df['Volume']
                data_df['TIMESTAMP_UTC'] = pd.to_datetime(data_df['TIMESTAMP_UTC'])
                data_df['hours'] = data_df['TIMESTAMP_UTC'].dt.hour
                summary = data_df[['hours','Volume','Multiply']].groupby(['hours']).sum()
                summary['Price_avg'] = summary['Multiply']/summary['Volume']
                std_agr = np.zeros((summary.shape[0]))
                for ind in range(summary.shape[0]):
                    hour = summary.index[ind]
                    std_agr[ind] = ((data_df.loc[data_df['hours']==hour,'Price'] - summary.loc[hour,'Price_avg'])**2).sum() / summary.loc[hour,'Price_avg']
                if hour_avg is None:
                    hour_avg = std_agr
                else:
                    hour_avg += std_agr
        vola_summary[stock] = (hour_avg/len(files)).tolist()
    return vola_summary
# TODO save as csv / maybe with minute window

def plot_vols(volume_summary:dict):
    fig = plt.figure(figsize=[16,9])
    ax = plt.subplot(111)

    for stock in volume_summary:
        ax.plot(volume_summary[stock], label=stock)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('Volume Distribution')
    plt.xlabel('hours')
    plt.ylabel('avg vol')
    plt.show()

if __name__=='__main__':
    stock_list = ['Adidas', 'Allianz','BASF','Bayer','BMW','Continental','Covestro','Covestro','Daimler','DeutscheBank','DeutscheBörse']
