# Model acording to: arXiv:2105.10430
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras as keras
from keras import backend as K

from data_handling.handle_data import get_file_names, get_raw_book, get_dirlist

def get_combined_book(stock_list:list):
    dir_list, path = get_dirlist(book=True)
    combined_books = []
    for stock in stock_list:
        combined_book = []
        files = get_file_names(stock, dir_list)
        for file in tqdm(files[:5]):
            book = get_raw_book(path+file)
            if not book.empty:
                book = book.dropna()
                book_np = book.to_numpy()
                combined_book.append(book_np)
        combined_np = np.concatenate(combined_book, axis=0)
        combined_books.append(combined_np)
    return combined_books

def label_and_normalize(data:np.array, label_steps=[100,300,500], classes=3):
    test_barrier = [0.00022, 0.00033, 0.00044]
    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    standardized = (data-mean)/std
    midpoints = np.mean(data[:,[0,2]], axis=1)
    labels = np.zeros((len(midpoints),len(label_steps),classes))  
    for index, steps in enumerate(label_steps):
        for i in tqdm(range(len(midpoints))):
            delta = (np.mean(midpoints[i+1:i+steps+1]/midpoints[i]-1))
            if delta >= test_barrier[index]/2.5:
                labels[i,index,0] = 1
            elif delta <= -test_barrier[index]/2.5:
                labels[i,index,2] = 1
            else:
                labels[i,index,1] = 1
    decoder_input_data = np.zeros((len(data), 1, 3))
    decoder_input_data[:, 0, 0] = 1.
    return standardized[:-label_steps[-1],:], labels[:-label_steps[-1],:,:], decoder_input_data[:-label_steps[-1],:,:]

class Data_Generator(tf.keras.utils.Sequence):
    
    def __init__(self, en_inputs, de_inputs, labels, batch_size, window, shuffle=True):
        
        self.en_inputs = en_inputs.copy()
        self.de_inputs = de_inputs.copy()
        self.labels = labels.copy()
        self.batch_size = batch_size
        self.window = window

        self.en_input_size = en_inputs.shape[1]
        self.de_input_size = de_inputs.shape[1:]
        self.labels_size = labels.shape[1:]
        self.shuffle = shuffle
        self.n = len(self.en_inputs)
        self.rand_ind = np.random.permutation((self.n-self.window)//self.window)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.rand_ind = np.random.permutation((self.n-self.window)//self.window)
    
    def __getitem__(self, index):
        start_ind = self.rand_ind[index:index+self.batch_size]
        x_1 = np.zeros((self.batch_size,self.window,self.en_input_size,1))
        x_2 = np.zeros((self.batch_size,)+self.de_input_size)
        y = np.zeros((self.batch_size,)+self.labels_size)
        for i, ind in enumerate(start_ind):
            ind_ = ind*self.window
            x_1[i,:,:,0] = self.en_inputs[ind_:ind_+self.window,:]
            x_2[i,:,:] = self.de_inputs[ind_+self.window-1,:,:]
            y[i,:,:] = self.labels[ind_+self.window-1,:,:]
        return [x_1,x_2], y
    
    def __len__(self):
        return (self.n // self.window) // self.batch_size

def get_model_attention(latent_dim, window, num_steps):
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
    convsecond_1 = keras.layers.Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = keras.layers.Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = keras.layers.MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = keras.layers.Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = keras.layers.Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(
        convsecond_output)

    # seq2seq
    encoder_inputs = conv_reshape
    encoder = keras.layers.CuDNNLSTM(latent_dim, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    # Set up the decoder, which will only process one timestep at a time.
    decoder_inputs = keras.layers.Input(shape=(1, 3))
    decoder_lstm = keras.layers.CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = keras.layers.Dense(3, activation='softmax', name='output_layer')

    all_outputs = []
    all_attention = []

    encoder_state_h = keras.layers.Reshape((1, int(state_h.shape[1])))(state_h)
    inputs = keras.layers.concatenate([decoder_inputs, encoder_state_h], axis=2)

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
    decoder_outputs = keras.layers.Lambda(lambda x: K.concatenate(x, axis=1), name='outputs')(all_outputs)
    decoder_attention = keras.layers.Lambda(lambda x: K.concatenate(x, axis=1), name='attentions')(all_attention)

    # Define and compile model as previously
    model = keras.models.Model([input_train, decoder_inputs], decoder_outputs)
    return model

if __name__=='__main__':
    data = get_combined_book(['Adidas','Daimler','BMW'])
    en_inputs, labels, de_inputs = label_and_normalize(data[0])
    split = int(len(en_inputs)*0.8)
    train_gen = Data_Generator(en_inputs[:split], de_inputs[:split], labels[:split], 32, 50)
    val_gen = Data_Generator(en_inputs[split:], de_inputs[split:], labels[split:], 32, 50)
    model = get_model_attention(32, train_gen.window, 3)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(train_gen, validation_data=val_gen, epochs=1)

    for data_ in data[1:]:
        en_inputs, labels, de_inputs = label_and_normalize(data_)
        split = int(len(en_inputs)*0.8)
        train_gen = Data_Generator(en_inputs[:split], de_inputs[:split], labels[:split], 32, 50)
        val_gen = Data_Generator(en_inputs[split:], de_inputs[split:], labels[split:], 32, 50)
        model.fit(train_gen, validation_data=val_gen, epochs=1)

'''
Last training:
Adidas
5000/5000 [==============================] - 116s 22ms/step - loss: 0.8183 - accuracy: 0.6047 - val_loss: 1.5291 - val_accuracy: 0.4488
Daimler
15440/15440 [==============================] - 336s 22ms/step - loss: 0.7020 - accuracy: 0.6868 - val_loss: 1.1749 - val_accuracy: 0.5487
BMW
13527/13527 [==============================] - 297s 22ms/step - loss: 0.6046 - accuracy: 0.7360 - val_loss: 1.1078 - val_accuracy: 0.6057

ACHTUNG: label means noch nicht gut verteilt -> varianz nutzen
'''
if False:
    import matplotlib.pyplot as plt
    ad_var = []
    dm_var = []
    db_var = []
    steps = 100
    for i in range(int(250000/steps)):
        ind = i*steps
        ad_tmp =np.mean(midpoints[ind+1:ind+steps+1]/midpoints[ind]-1)
        dm_tmp =np.mean(midpoints2[ind+1:ind+steps+1]/midpoints2[ind]-1) 
        db_tmp =np.mean(midpoints3[ind+1:ind+steps+1]/midpoints3[ind]-1)
        ad_var.append(ad_tmp)
        dm_var.append(dm_tmp)
        db_var.append(db_tmp)
    plt.hist(ad_var, bins=20)
    plt.show()
    plt.hist(dm_var, bins=20)
    plt.show()
    plt.hist(db_var, bins=20)
    plt.show()