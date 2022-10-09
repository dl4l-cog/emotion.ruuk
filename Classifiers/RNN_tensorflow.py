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
import collections
import pathlib
import tensorflow as tf
#import tensorflow_addons as tfa

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization

train = load_dataset('emotion', split='train')
test  = load_dataset('emotion', split='test')

train_dataset = train.to_tf_dataset(columns=["text"], label_cols=["label"],  batch_size = 2)
test_dataset = test.to_tf_dataset(columns=["text"], label_cols=["label"], batch_size = 2)


#for alltexts, alllabels in train_dataset.take(8000):
#  #print('text: ', example.numpy())
#  print("hi")


BUFFER_SIZE = 10000
BATCH_SIZE = 32
seed = 32

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))
vocab = np.array(encoder.get_vocabulary())



#encoded_example = encoder(example)[:3].numpy()
#encoded_sequences = encoder(alltexts).numpy()
#print(encoded_sequences)
#print(encoded_example)
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6, activation='softmax')
])


model = tf.keras.models.load_model('model_keras')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


model.summary()
"""
history = model.fit(train_dataset, epochs=4,
                    validation_data=test_dataset,
                    validation_steps=30)

model.save("model_keras")
"""
test_loss, test_acc = model.evaluate(test_dataset)



print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

sample_text = ('I am not angry')
predictions = model.predict(np.array([sample_text]))
#metric = tfa.metrics.F1Score(num_classes=6, threshold=0.5)

print(predictions)
print('Sadness: {:.4f} || Joy: {:.4f} || Love: {:.4f} || Anger: {:.4f} || Fear: {:.4f} || Suprise: {:.4f}'.format(predictions[0][1], predictions[0][1], predictions[0][2], predictions[0][3], predictions[0][4], predictions[0][5]))









"""
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

x = np.zeros(shape=(len(train_text), maxlen, 2000), dtype=torch.int64)
y = torch.tensor(train_label, dtype=torch.int64)
wordmap = {c:i for i, c in enumerate(dictionary, start=2)}

for i in range(len(train_text)):
    splitted_text = train_text[i].split()
    for j in range(len(splitted_text)):
        #print(wordmap[" "])
        if wordmap[splitted_text[j]] < 2000:
            x[i, j, wordmap[splitted_text[j]]] = 1

# Build a very simple recurrent neural network model
m = tf.keras.Sequential((
        tf.keras.layers.SimpleRNN(8, return_sequences=True),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ))
m.compile(optimizer='adam', loss='binary_crossentropy')

m.build((None, None, len(wordmap)))
m.summary()

m.fit(x, y, batch_size = 1, epochs=20, verbose=1)



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