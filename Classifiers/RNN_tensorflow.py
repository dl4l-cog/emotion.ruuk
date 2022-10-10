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
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(6, activation='softmax')
])


#model = tf.keras.models.load_model('model_keras')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


model.summary()

history = model.fit(train_dataset, epochs=4,
                    validation_data=test_dataset,
                    validation_steps=30)

model.save("model_keras")

test_loss, test_acc = model.evaluate(test_dataset)



print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

sample_text = ('I am not angry')
predictions = model.predict(np.array([sample_text]))
#metric = tfa.metrics.F1Score(num_classes=6, threshold=0.5)

print(predictions)
print('Sadness: {:.4f} || Joy: {:.4f} || Love: {:.4f} || Anger: {:.4f} || Fear: {:.4f} || Suprise: {:.4f}'.format(predictions[0][1], predictions[0][1], predictions[0][2], predictions[0][3], predictions[0][4], predictions[0][5]))








