#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import numpy as np
import tensorflow as tf
import json
from keras import backend as K
import re


# load Datasets
train = load_dataset('emotion', split='train')
test  = load_dataset('emotion', split='test')

# Convert Huggingfacedata into Tensorflow Data
train_dataset = train.to_tf_dataset(columns=["text"], label_cols=["label"], batch_size = 64)
test_dataset = test.to_tf_dataset(columns=["text"], label_cols=["label"], batch_size = 64)


#####################################################################


# define evaluation functions
#--------------------------------------------------------------------

def recall_m(y_true, y_pred):
    true_positives = K.sum( K.round( K.clip( y_true * y_pred, 0, 1)))
    possible_positives = K.sum( K.round( K.clip( y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum( K.round( K.clip( y_true * y_pred, 0, 1)))
    predicted_positives = K.sum( K.round( K.clip( y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision*recall) / (precision + recall + K.epsilon()))


###################################################################


# Select the 10000 most frequent words from the training data and give each word a number
VOCAB_SIZE = 4000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))
# Vocab is the dictionary of the 10000 most frequent words
vocab = np.array(encoder.get_vocabulary())

model = tf.keras.Sequential([
    encoder,
    # Embedding Turns numbers into One-hot-vectors compatible with the NN
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    # Bidirectional LSTM Layers integrate a long and short-term memory into the NN
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True, recurrent_dropout=0.2)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, recurrent_dropout=0.2, activity_regularizer=tf.keras.regularizers.L2(0.01))),
    # Output layer consists of 6 different emotions
    tf.keras.layers.Dense(6, activation='softmax')
])

# This will load the already trained model
#model = tf.keras.models.load_model('model_keras_big')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc', f1_m, precision_m, recall_m]
)

model.summary()

# Training the model
EPOCHS  = 10
history = model.fit(train_dataset, epochs=EPOCHS,
                    validation_data=test_dataset, validation_steps=30)

# Saving the model so it doesn't have to be trained every time
model.save("model_keras_big")

# print evaluation results
#--------------------------------------------------------------------
loss, accuracy, f1_score, precision, recall = model.evaluate(test_dataset)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
print('Test F1', f1_score)
print('precision', precision)
print('recall', recall)