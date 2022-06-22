# Model acording to: arXiv:2105.10430
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras as keras
from keras import backend as K
import tensorflow_addons as tfa

#from data_handling.handle_data import get_file_names, get_raw_book, get_dirlist, get_trades
from handle_data import get_file_names, get_raw_book, get_dirlist, get_trades

def compress_book(data: pd.DataFrame, timedelta: pd.Timedelta = pd.Timedelta(microseconds=10000)):
    data = data.copy().reset_index()
    data = data.resample(timedelta, on='TIMESTAMP_UTC').mean()
    return data.dropna()


def create_trade_list(stock, days=59):
    timedelta = pd.Timedelta(microseconds=10000)
    dir_list, path = get_dirlist(book=True)
    dir_list_t, path_t = get_dirlist(book=False)
    files = get_file_names(stock, dir_list)
    files_t = get_file_names(stock, dir_list_t)
    trade_list = []
    for i in tqdm(range(len(files[:days]))):
        book = get_raw_book(path+files[i])
        if not book.empty:
            trades = get_trades(path_t+files_t[i])
            midpoints = book.iloc[:, [0, 2]].mean(axis=1)
            for i in range(len(trades)):
                ind = midpoints.index[midpoints.index < trades.index[i]][-1]
                trades.iloc[i, 0] = (
                    trades.iloc[i, 0]-midpoints[ind])/midpoints[ind]
            data_abs = trades.abs()
            data = trades.copy().reset_index()
            data_abs = data_abs.reset_index()
            data1 = (data.resample(timedelta, on='TIMESTAMP_UTC').max())[
                'Price'] < 0
            data2 = (data_abs.resample(
                timedelta, on='TIMESTAMP_UTC').max())['Price']
            data2[data1] = data2[data1] * (-1)
            trade_list.append(data2.dropna())
    return pd.concat(trade_list)


def get_combined_book(stock_list: list, days=10) -> list[pd.DataFrame]:
    dir_list, path = get_dirlist(book=True)
    combined_books = []
    for stock in stock_list:
        combined_book = []
        files = get_file_names(stock, dir_list)
        for file in tqdm(files[:days]):
            book = get_raw_book(path+file)
            if not book.empty:
                book = compress_book(
                    book, timedelta=pd.Timedelta(microseconds=100000))
                book = book.dropna()
                combined_book.append(book)
        combined_books.append(pd.concat(combined_book, axis=0))
    return combined_books


def get_quantile(data, steps=30):
    midpoints = np.mean(data[:, [0, 2]], axis=1)
    midpoints = midpoints[1:]/midpoints[:-1]
    change = np.zeros((len(midpoints)-steps))
    for i in range(len(midpoints)-steps):
        change[i] = np.mean(np.cumprod(midpoints[i:i+steps])-1)
    return np.quantile(change, 0.33), np.quantile(change, 0.66)


def label_and_normalize(data: np.array, label_steps=[200, 400, 500, 700, 1000], classes=3, window=500):
    #barrier = []
    # for steps in label_steps:
    #    barrier.append(get_quantile(data,steps=steps))
    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    standardized = (data-mean)/std
    midpoints = np.mean(data[:, [0, 2]], axis=1)
    labels = np.zeros((len(midpoints), len(label_steps), classes))
    decoder_input_data = np.zeros((len(data), 1, classes))
    for i in tqdm(range(len(midpoints)-label_steps[-1])):
        for index, steps in enumerate(label_steps):
            delta = np.mean(midpoints[i+1:i+steps+1]/midpoints[i]-1)
            if delta >= 0.0004:
                labels[i, index, 0] = 1.
            elif delta <= -0.0004:
                labels[i, index, 2] = 1.
            else:
                labels[i, index, 1] = 1.
        decoder_delta = np.mean(
            midpoints[max(i-window, 0):i+1]/midpoints[max(i-window, 0)]-1)
        if decoder_delta >= 0.0004:
            decoder_input_data[i, 0, 0] = 1.
        elif decoder_delta <= -0.0004:
            decoder_input_data[i, 0, 2] = 1.
        else:
            decoder_input_data[i, 0, 1] = 1.

    return standardized[:-label_steps[-1], :], labels[:-label_steps[-1], :, :], decoder_input_data[:-label_steps[-1], :, :]


def label_up_down(data: np.array, label_steps=[200, 400, 500, 700, 1000], classes=2, window=500):
    midpoints = np.mean(data[:, [0, 2]], axis=1)
    labels = np.zeros((len(midpoints), len(label_steps), classes))
    decoder_input_data = np.zeros((len(data), 1, classes))
    for i in tqdm(range(len(midpoints)-label_steps[-1])):
        for index, steps in enumerate(label_steps):
            delta = np.mean(midpoints[i+1:i+steps+1]/midpoints[i]-1)
            if delta > 0.0:
                labels[i, index, 0] = 1.
            else:
                labels[i, index, 1] = 1.
        decoder_delta = np.mean(
            midpoints[max(i-window, 0):i+1]/midpoints[max(i-window, 0)]-1)
        if decoder_delta > 0.0:
            decoder_input_data[i, 0, 0] = 1.
        else:
            decoder_input_data[i, 0, 1] = 1.
    return data[:-label_steps[-1], :].copy(), labels[:-label_steps[-1], :, :], decoder_input_data[:-label_steps[-1], :, :]


def label_exec_prob(book: pd.DataFrame, trades: pd.DataFrame, label_steps=[200, 400, 500, 700, 1000], window=500):
    std = np.std(book, axis=0)
    mean = np.mean(book, axis=0)
    standardized = (book-mean)/std
    midpoints = np.mean(book.iloc[:, [0, 2]], axis=1)
    labels = np.zeros((len(midpoints), len(label_steps), 1))
    decoder_input_data = np.zeros((len(book), 2, 1))
    levels = np.random.choice(np.arange(0, 39, 2), size=(len(book)))
    for i in tqdm(range(len(midpoints)-label_steps[-1])):
        pos = (book.iloc[i, levels[i]] - midpoints[i]) / midpoints[i]
        for index, steps in enumerate(label_steps):
            interval = trades[(trades.index > book.index[i]) & (
                trades.index < book.index[i+steps+1])]
            if pos > 0.0:
                if interval.max() > pos:
                    labels[i, index, 0] = 1
            else:
                if interval.min() < pos:
                    labels[i, index, 0] = 1
        interval2 = trades[(trades.index > book.index[max(
            i-window, 0)]) & (trades.index < book.index[i])]
        if pos > 0.0:
            if interval2.max() > pos:
                decoder_input_data[i, 0, 0] = 1
        else:
            if interval2.min() < pos:
                decoder_input_data[i, 0, 0] = 1
        decoder_input_data[i, 1, 0] = pos

    # new try achtung noch nicht clean bei labeling indexing
    for i in tqdm(range(len(trades))):
        ind = book.index.get_loc(
            book.index[book.index <= trades.index[i]][-1:])
        for index, steps in enumerate(label_steps):
            ind_range = np.arange(ind+1-100, ind+1)
            pos = (book.to_numpy()[ind_range, levels[ind_range]
                                   ] - midpoints[ind_range]) / midpoints[ind_range]
            if trades.iloc[i, 0] > 0:
                labeling = pos < trades.iloc[i, 0]
                labels[labeling] = 1.
            else:
                labeling = pos > trades.iloc[i, 0]
                labels[labeling] = 1.
    return standardized[:-label_steps[-1], :], labels[:-label_steps[-1], :, :], decoder_input_data[:-label_steps[-1], :, :]


def label_intensity(data: np.array, label_steps=[200, 400, 500, 700, 1000], window=500):
    midpoints = np.mean(data[:, [0, 2]], axis=1)
    labels = np.zeros((len(midpoints), len(label_steps), 1))
    decoder_input_data = np.zeros((len(data), 1, 1))
    for i in tqdm(range(len(midpoints)-label_steps[-1])):
        decoder_delta = np.std(
            midpoints[max(i-window, 0):i+1]/midpoints[max(i-window, 0)]-1)
        decoder_input_data[i, 0, 0] = decoder_delta
        for index, steps in enumerate(label_steps):
            delta = abs(np.mean(midpoints[i+1:i+steps+1]/midpoints[i]-1))
            if decoder_delta > 0:
                labels[i, index, 0] = min(delta/(2*decoder_delta),1)
            else:
                labels[i, index, 0] = 0
    return data[:-label_steps[-1], :].copy(), labels[:-label_steps[-1], :, :], decoder_input_data[:-label_steps[-1], :, :]


class Data_Generator(tf.keras.utils.Sequence):

    def __init__(self, en_inputs, de_inputs, labels, batch_size, window, overlap=2, shuffle=True):

        self.en_inputs = en_inputs.copy()
        self.de_inputs = de_inputs.copy()
        self.labels = labels.copy()
        self.batch_size = batch_size
        self.window = window
        self.overlap = overlap

        self.en_input_size = en_inputs.shape[1]
        self.de_input_size = de_inputs.shape[1:]
        self.labels_size = labels.shape[1:]
        self.shuffle = shuffle
        self.n = len(self.en_inputs)
        if self.shuffle:
            self.rand_ind = np.random.permutation(
                int((self.n-self.window)/(self.window/self.overlap)))
        else:
            self.rand_ind = np.arange(int((self.n-self.window)/(self.window/self.overlap)))

    def on_epoch_end(self):
        if self.shuffle:
            self.rand_ind = np.random.permutation(
                int((self.n-self.window)/(self.window/self.overlap)))
        else:
            self.rand_ind = np.arange(int((self.n-self.window)/(self.window/self.overlap)))

    def normalize_window(self, input):
        # normalize with min max in window
        data = input.copy()
        data[:,0::2] = data[:,0::2]-np.min(data[:,0::2])
        data[:,1::2] = data[:,1::2]-np.min(data[:,1::2])
        data[:,0::2] = data[:,0::2]/np.max(data[:,0::2])
        data[:,1::2] = data[:,1::2]/np.max(data[:,1::2])
        return data

    def __getitem__(self, index):
        start_ind = self.rand_ind[index:index+self.batch_size]
        x_1 = np.zeros((self.batch_size, self.window, self.en_input_size, 1))
        x_2 = np.zeros((self.batch_size,)+self.de_input_size)
        y = np.zeros((self.batch_size,)+self.labels_size)
        for i, ind in enumerate(start_ind):
            ind_ = ind * int(self.window/self.overlap)
            x_1[i, :, :, 0] = self.normalize_window(self.en_inputs[ind_:ind_+self.window, :])
            x_2[i, :, :] = self.de_inputs[ind_+self.window-1, :, :]
            y[i, :, :] = self.labels[ind_+self.window-1, :, :]
        return [x_1, x_2], y

    def __len__(self):
        return int((self.n-self.window)/(self.window/self.overlap))


def get_model_attention(latent_dim, window, num_steps, classes=3, fin_act='softmax'):
    # Luong Attention
    # https://arxiv.org/abs/1508.04025

    input_train = keras.layers.Input(shape=(window, 40, 1))

    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(input_train)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = keras.layers.Conv2D(
        64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = keras.layers.Conv2D(
        64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = keras.layers.MaxPooling2D(
        (3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = keras.layers.Conv2D(
        64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate(
        [convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = keras.layers.Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(
        convsecond_output)

    # seq2seq
    encoder_inputs = conv_reshape
    encoder = keras.layers.CuDNNLSTM(
        latent_dim, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    # Set up the decoder, which will only process one timestep at a time.
    decoder_inputs = keras.layers.Input(shape=(1, classes))
    decoder_lstm = keras.layers.CuDNNLSTM(
        latent_dim, return_sequences=True, return_state=True)
    decoder_dense = keras.layers.Dense(classes,
                                        activation=fin_act, 
                                        name='output_layer')

    all_outputs = []
    all_attention = []

    encoder_state_h = keras.layers.Reshape((1, int(state_h.shape[1])))(state_h)
    inputs = keras.layers.concatenate(
        [decoder_inputs, encoder_state_h], axis=2)

    for _ in range(num_steps):
        # h'_t = f(h'_{t-1}, y_{t-1}, c)
        outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
        # dot
        attention = keras.layers.dot([outputs, encoder_outputs], axes=2)
        attention = keras.layers.Activation('softmax')(attention)
        # context vector
        context = keras.layers.dot([attention, encoder_outputs], axes=[2, 1])
        context = keras.layers.BatchNormalization(momentum=0.6)(context)

        # y = g(h'_t, c_t)
        decoder_combined_context = keras.layers.concatenate([context, outputs])
        outputs = decoder_dense(decoder_combined_context)
        all_outputs.append(outputs)
        all_attention.append(attention)

        inputs = keras.layers.concatenate([outputs, context], axis=2)
        states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = keras.layers.Lambda(
        lambda x: K.concatenate(x, axis=1), name='outputs')(all_outputs)
    decoder_attention = keras.layers.Lambda(
        lambda x: K.concatenate(x, axis=1), name='attentions')(all_attention)

    # Define and compile model as previously
    model = keras.models.Model([input_train, decoder_inputs], decoder_outputs)
    return model

def score_midpoint(model1, model2, en_input, de_inputs1, labels1, de_inputs2, labels2, split):
    midpoints = np.mean(en_input[split:, [0, 2]], axis=1)
    val_gen1 = Data_Generator(en_inputs[split:], de_inputs1[split:], labels1[split:], batch_size, window_size, overlap=1, shuffle=False)
    val_gen2 = Data_Generator(en_inputs[split:], de_inputs2[split:], labels2[split:], batch_size, window_size, overlap=1, shuffle=False)
    pred1 = model1.predict(val_gen1)
    pred2 = model2.predict(val_gen2)
    score = (pred1[:,4,0]-pred1[:,4,1])*pred2[:,4,0]

    return score, midpoints



if __name__ == '__main__':
    # Adidas, 'Allianz','BASF' 'Bayer', 'BMW', 'Continental','Covestro', 'Covestro', 'Daimler', 'DeutscheBank', 'DeutscheBörse'
    mode = 'direction'
    stock_list = ['DeutscheBank', ]
    window_size = 200
    batch_size = 32
    if mode=='direction':
        model = get_model_attention(32, window_size, 5, classes=2)
        model.compile(loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(
        ), keras.metrics.Recall(), tf.keras.metrics.Precision()], optimizer='adam')    
    else:
        model = get_model_attention(32, window_size, 5, classes=1, fin_act='relu')
        model.compile(loss='mean_absolute_error', optimizer='adam')
    data = get_combined_book(stock_list, days=10)

    for _ in range(1):
        for data_ in data:
            if mode=='direction':
                en_inputs, labels, de_inputs = label_up_down(data_.to_numpy())
            else:
                en_inputs, labels, de_inputs = label_intensity(data_.to_numpy())
            print(np.mean(labels, axis=0))
            split = int(len(en_inputs)*0.8)
            train_gen = Data_Generator(
                en_inputs[:split], de_inputs[:split], labels[:split], batch_size, window_size, overlap=1)
            val_gen = Data_Generator(
                en_inputs[split:], de_inputs[split:], labels[split:], batch_size, window_size, overlap=1)
            model.fit(train_gen, validation_data=val_gen, epochs=3)

'''
Training history:
variable barrier
val acc between 0.4 , 0.27 :/
fixed barriers per stock may be better

Last training:
Adidas
5000/5000 [==============================] - 116s 22ms/step - loss: 0.8183 - accuracy: 0.6047 - val_loss: 1.5291 - val_accuracy: 0.4488
Daimler
15440/15440 [==============================] - 336s 22ms/step - loss: 0.7020 - accuracy: 0.6868 - val_loss: 1.1749 - val_accuracy: 0.5487
BMW
13527/13527 [==============================] - 297s 22ms/step - loss: 0.6046 - accuracy: 0.7360 - val_loss: 1.1078 - val_accuracy: 0.6057

ACHTUNG: label means noch nicht gut verteilt -> varianz nutzen
'''

# crap
if False:

    books = get_combined_book(['Adidas', 'Daimler', 'BMW'])
    for steps in [30, 100, 180]:
        for book in books:
            print(get_quantile(book.copy(), steps=steps))

    data1 = books[0].copy()
    data2 = books[1].copy()
    data3 = books[2].copy()
    midpoints1 = np.mean(data1[:, [0, 2]], axis=1)
    midpoints2 = np.mean(data2[:, [0, 2]], axis=1)
    midpoints3 = np.mean(data3[:, [0, 2]], axis=1)
    midpoints1 = midpoints1[1:]/midpoints1[:-1]-1
    midpoints2 = midpoints2[1:]/midpoints2[:-1]-1
    midpoints3 = midpoints3[1:]/midpoints3[:-1]-1
    import matplotlib.pyplot as plt
    ad_var = []
    dm_var = []
    db_var = []
    steps = 100
    for i in range(int(250000/steps)):
        ind = i*steps
        ad_tmp = np.mean(midpoints1[ind+1:ind+steps+1]/midpoints1[ind]-1)
        dm_tmp = np.mean(midpoints2[ind+1:ind+steps+1]/midpoints2[ind]-1)
        db_tmp = np.mean(midpoints3[ind+1:ind+steps+1]/midpoints3[ind]-1)
        ad_var.append(ad_tmp)
        dm_var.append(dm_tmp)
        db_var.append(db_tmp)
    plt.hist(ad_var, bins=20)
    plt.show()
    plt.hist(dm_var, bins=20)
    plt.show()
    plt.hist(db_var, bins=20)
    plt.show()
