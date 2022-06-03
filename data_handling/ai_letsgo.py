import numpy as np
import pandas as pd
import os
import unicodedata
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from keras.models import Sequential
from keras.layers import Input, CuDNNLSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
import tensorflow as tf
import keras
import keras.backend as K
from data_handling.handle_data import get_file_names, get_raw_book, get_midpoints


def plot_book(book_state:pd.Series):
    book_np = book_state.to_numpy()
    bid_vol = np.flip(book_np[np.arange(1,40,4)])
    bid_pr = np.flip(book_np[np.arange(0,40,4)])
    ask_vol = book_np[np.arange(3,40,4)]
    ask_pr = book_np[np.arange(2,40,4)]
    plt.bar(np.concatenate((bid_pr,ask_pr)),np.concatenate((bid_vol,ask_vol)))
    plt.show()

def transform_minutes(book_states):
    in_time = True
    timestamp = pd.Timestamp(2021,book_tmp.index[0].month,book_tmp.index[0].day,8,0)
    minutes = np.zeros((int(60*8.5),40))
    ind = 0
    while in_time:
        tmp_time = timestamp + pd.DateOffset(minutes = 1)
        minutes[ind,:] = book_states[(book_states.index > timestamp) & (book_states.index < tmp_time)].mean()
        timestamp = tmp_time
        ind += 1
        if tmp_time > pd.Timestamp(2021,book_tmp.index[0].month,book_tmp.index[0].day,16,29):
            in_time = False
    return minutes

def build_training(mins):
    window = 20
    label_range = 3
    x_vals = np.zeros((mins.shape[0]-window-label_range,window,mins.shape[1]))
    y_vals = np.zeros((mins.shape[0]-window-label_range,1))
    for i in range(1,x_vals.shape[0]):
        x_vals[i-1,:,:] = mins[i:i+window,:].copy() / mins[i-1:i-1+window,:].copy() -1
        y_vals[i-1,:] = np.mean(mins[i+window+label_range-1,[0,2]]) > np.mean(mins[i+window-1,[0,2]])
    x_vals[np.isnan(x_vals)] = 0
    return x_vals, y_vals

def check_time_window():
    book_test = get_raw_book(path_book+files[0])
    in_time = True
    timestamp = pd.Timestamp(2021,1,5,8,0)
    while in_time:
        tmp_time = timestamp + pd.DateOffset(minutes = 1)
        plot_book(book_test[ (book_test.index > timestamp) & (book_test.index < tmp_time)].mean())
        timestamp = tmp_time
        if tmp_time > pd.Timestamp(2021,1,5,8,25):
            in_time = False

stock_list = ['Adidas', 'Allianz','BASF','Bayer','BMW','Continental','Covestro','Covestro','Daimler','DeutscheBank','DeutscheBörse']
path_trades = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/trades/'
path_book = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/book/'
dir_list_t = [unicodedata.normalize('NFC', f) for f in os.listdir(path_trades)]
dir_list_b = [unicodedata.normalize('NFC', f) for f in os.listdir(path_book)]

x_data = None
y_data = None
for stock in stock_list:
    files = get_file_names(stock, dir_list_b)
    for file in tqdm(files):
        book_tmp = get_raw_book(path_book+file)
        if not book_tmp.empty:
            minutes = transform_minutes(book_tmp)
            out = build_training(minutes)
            if x_data is None:
                x_data = out[0]
                y_data = out[1]
            else:
                x_data = np.concatenate((x_data,out[0]), axis=0)
                y_data = np.concatenate((y_data,out[1]), axis=0)

def everknowing_entity(timestamp:pd.Timestamp, stock:str, steps:int) -> int:
    path_book = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/book/'
    dir_list_b = [unicodedata.normalize('NFC', f) for f in os.listdir(path_book)]
    files = get_file_names(stock, dir_list_b)
    midpoints = get_midpoints(path_book+files[timestamp.day-1])
    ind_0 = midpoints.index[midpoints.index > timestamp][0]
    ind_n = midpoints.index[midpoints.index > timestamp][steps]
    vola = midpoints[(midpoints.index < timestamp) & (midpoints.index > timestamp-pd.Timedelta(minutes=10))].std()
    if midpoints[ind_0] + vola < midpoints[ind_n]:
        up = 1
    elif midpoints[ind_0] - vola > midpoints[ind_n]:
        up = -1
    else:
        up = 0
    return up

def lstm_model():
    model = Sequential()
    model.add(Dense(64,activation='tanh'))
    model.add(Dropout(0.3,noise_shape=(None,20,64)))
    model.add(Bidirectional(CuDNNLSTM(64)))
    model.add(Dense(64,activation='tanh'))
    model.add(Dense(1,activation='sigmoid', kernel_regularizer=keras.regularizers.L1L2(l1=1e-8, l2=1e-8),
                                        bias_regularizer=keras.regularizers.L2(1e-8),
                                        activity_regularizer=keras.regularizers.L2(1e-8)))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy() , optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=tf.keras.metrics.BinaryAccuracy())

    model.fit(x_data, y_data, batch_size=8, epochs=4, validation_split=0.1)

# last result acc=58, val_acc=60
# multi stock predictor?