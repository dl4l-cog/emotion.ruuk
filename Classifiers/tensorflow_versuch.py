#!/usr/bin/env python3
# run using 'python baseline.py'
"""A simple B/I segmenter using RNNs as starter code / baseline for a3.
"""
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import torch
import torch.nn as nn
from keras.preprocessing import text
from keras.preprocessing import sequence
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM


train = load_dataset('emotion', split='train')
test  = load_dataset('emotion', split='test')
def convert(data_from_dict, split):
    list = []
    for i in range(len(data_from_dict)):
        list.append(data_from_dict[i][split])
    return list

train_text = convert(train, 'text')
train_label = convert(train, 'label')
test_text = convert(test, 'text')
test_label = convert(test, 'label')



def construct_dictionary(train_text):
    dictionary = {"Hello"}
    for text in train_text:
        list_of_words = text.split()
        for word in list_of_words:
            dictionary.add(word)
    return dictionary

def max_length(list_of_texts):
    maxlen = 0
    for text in list_of_texts:
        text_length = len(text.split())
        if text_length > maxlen:
            maxlen = text_length
    return maxlen

maxlen = max_length(train_text)
dictionary = construct_dictionary(train_text)

print(max_length(train_text))

x = np.zeros(shape=(len(train_text), maxlen, 2000), dtype=int)
y = torch.tensor(train_label)
wordmap = {c:i for i, c in enumerate(dictionary, start=2)}

for i in range(len(train_text)):
    splitted_text = train_text[i].split()
    for j in range(len(splitted_text)):
        #print(wordmap[" "])
        if wordmap[splitted_text[j]] < 2000:
            x[i, j, wordmap[splitted_text[j]]] = 1

# Build a very simple recurrent neural network model
m = tf.keras.Sequential((
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ))

m.compile(optimizer='adam', loss='binary_crossentropy')

m.build((None, None, len(wordmap)))
m.summary()

m.fit(x, y, epochs=20, verbose=1)

"""

charmap = {c:i for i, cin enumerate(chars, start=2)}
charmap['pad'], charmap['unk'] = 0, 1

x = np.zeros(shape=(len(words), maxlen, len(charmap)), dtype=int)
y = np.zeros(shape=(len(words), maxlen), dtype=int)
for i in range(len(words)):
    for j, c in enumerate(words[i]):
        x[i, j, charmap[c]] = 1
        y[i, j] = labels[i][j]

# Build a very simple recurrent neural network model
m = tf.keras.Sequential((
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ))

m.compile(optimizer='adam', loss='binary_crossentropy')

m.build((None, None, len(charmap)))
m.summary()

m.fit(x, y, epochs=20, verbose=1)

# The following is only for the demonstration of the evaluation.
# Remember that evaluating the model on training set is not really
# useful in practice.

train_pred = m.predict(x)
pred_words = []
for i, w in enumerate(words):
    morphs, start = [], 0
    for split in np.where(train_pred[i].flatten() > 0.5)[0]:
        if split == 0: continue # skip the 'B' prediction at the beginning
        morphs.append(''.join(w[start:split]))
        start = split
    morphs.append(''.join(w[start:])) # last segment
    pred_words.append(morphs)

tp, fp, fn = 0, 0, 0
for i, w in enumerate(words_split):
    # this is not strictly correct in case there are repeated morphemes
    gold, pred = set(w), set(pred_words[i])
    tp += len(gold & pred)
    fp += len(pred - gold)
    fn += len(gold - pred)
p = 0 if tp == 0 else tp / (tp + fp)
r = 0 if tp == 0 else tp / (tp + fn)
f = 0 if p == 0 else 2*p*r / (p + r)
print(f"Training set scores (p/r/f): {p:.04f} / {r:.04f} / {f:.04f}")


# Read the test set, encode the same way
testwords = []
with open('dll-a3.test', 'rt') as f:
    for line in f:
        testwords.append(line.strip()[:maxlen])

test_x = np.zeros(shape=(len(testwords), maxlen, len(charmap)), dtype=int)
for i in range(len(testwords)):
    for j, c in enumerate(testwords[i]):
        test_x[i, j, charmap.get(c, charmap['unk'])] = 1

pred = m.predict(test_x)

# 'pred' now contains probability of a split point before each
# character 
# We can determine where to insert spaces, in the test words, and
# write it to the prediction file.

with open('a3-baseline.predictions', 'wt') as outf:
    for i, w in enumerate(testwords):
        print(w[0], file=outf, end="")
        for j, c in enumerate(w[1:], start=1): # skip the first prediction
            if pred[i,j] > 0.5:
                print(' ', file=outf, end="")
            print(c, file=outf, end="")
        print(file=outf)
"""