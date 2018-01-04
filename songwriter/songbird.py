from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
import numpy as np
import pandas as pd
import random
import sys

model = None
maxlen = 40
char_indices = None
chars = None
indices_char = None
PATH = '/mnt/shared/rabbiteer2017/songgenerator.h5'

def init():
    global model
    global chars
    global char_indices
    global indices_char
    global maxlen

    path = '/mnt/shared/rabbiteer2017/data/songdata.csv'
    df = pd.read_csv(path)
    #df = df[df['artist']=='The Beatles']
    text = df['text'].str.cat(sep='\n').lower()
    # Output the length of the corpus
    print('corpus length:', len(text))


    # Create a sorted list of the characters
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))

    # Corpus is going to take tooooo long to train, so lets make it shorter
    text = text[:1000000]
    print('truncated corpus length:', len(text))

    # Create a dictionary where given a character, you can look up the index and vice versa
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    print(len(char_indices))
    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40

    # Load model
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print("Compiling model complete...")

    model = load_model(PATH)
    print(model)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def gettext(sentence):
    global model
    global chars
    global char_indices
    global indices_char
    global maxlen
    # 1 hot encode
    sentence = sentence.lower()
    sentence = sentence[:maxlen]
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    print(model.predict(x, verbose=0)[0])
    print(sum(model.predict(x, verbose=0)[0]))

    # Predict the next 400 characters based on the seed
    generated = ''
    original = sentence
    for i in range(800):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.2)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return original + generated
