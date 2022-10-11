#!/usr/bin/env python3
# run using 'python baseline.py'
"""A simple B/I segmenter using RNNs as starter code / baseline for a3.
"""
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization
import json
from keras import backend as K
import re



train = load_dataset('emotion', split='train')
test  = load_dataset('emotion', split='test')

# Convert Huggingfacedata into Tensorflow Data
train_dataset = train.to_tf_dataset(columns=["text"], label_cols=["label"],  batch_size = 64)
test_dataset = test.to_tf_dataset(columns=["text"], label_cols=["label"], batch_size = 64)



######################## EVALUATION #######################
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
###################################################################



# Select the 10000 most frequent words from the training data and give each word a number
VOCAB_SIZE = 4000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
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
    metrics=['acc', f1_m,precision_m, recall_m]
)


model.summary()
# Training the model

history = model.fit(train_dataset, epochs=5,
                    validation_data=test_dataset,
                    validation_steps=30)
# Saving the model so it does'nt have to be trained every time
model.save("model_keras_big")


loss, accuracy, f1_score, precision, recall = model.evaluate(test_dataset)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
print('Test F1', f1_score)
print('precision', precision)
print('recall', recall)



# Sample Texts to test the model
sample_text = ('''Im so sick of this bullshit''')
predictions = model.predict(np.array([sample_text]))
#metric = tfa.metrics.F1Score(num_classes=6, threshold=0.5)
print('Sadness: {:.4f} || Joy: {:.4f} || Love: {:.4f} || Anger: {:.4f} || Fear: {:.4f} || Suprise: {:.4f}'.format(predictions[0][0], predictions[0][1], predictions[0][2], predictions[0][3], predictions[0][4], predictions[0][5]))

def text_to_list_of_sentences(text_list):
    text_list = text_list.split('.')
    return text_list



## Sums ##
def sum_over_parts(list_of_parts):
    sums_of_emotions = []
    for emotion in range(6):
        sum = 0
        for part in list_of_parts:
            sum = sum + part[emotion]
        sums_of_emotions.append(sum)
    overall_sum = 0
    for sum in sums_of_emotions:
        overall_sum = overall_sum + sum
    for i in range(6):
        sums_of_emotions[i] = round(sums_of_emotions[i] / overall_sum, 4)
            
    return sums_of_emotions

#print(sum_over_parts(dectected_emotions))
# detected emotions ist Matrix mit evaluierten Emotionen
# 1 = Sadness, 2 = Joy, 3 = Love, 4 = Anger, 5 = Fear, 6 = Suprise

# Tweets einlesen

with open('test.jsonl', 'r') as json_file:
    json_list = list(json_file)


# Wandelt Json in Liste an Texten um
def json_to_listOfTexts(json_list):        
    tweet_text_list = []
    for tweets in json_list:
        result = json.loads(tweets)
        print(result)
        removeUrl(result)
        removeHashtag(result)
        removeAT(result)
        if result["lang"] == "en" and islongerthanthreewords(result) and not(result["is_quote_status"]) and not(isReply(result)): 
            tweet_text_list.append(result["full_text"])

def removeUrl(tweet):
    tweet["full_text"] = re.sub(r" http\S+", "", tweet["full_text"])

def removeHashtag(tweet):
    tweet["full_text"] = re.sub(r" #\S+", "", tweet["full_text"])

def removeAT(tweet):
    tweet["full_text"] = re.sub(r" @\S+", "", tweet["full_text"])

def islongerthanthreewords(tweet):
    list_of_words = tweet["full_text"].split()
    return len(list_of_words) > 3

def isReply(tweet):
    return tweet["in_reply_to_status_id"] is not None

def predictEmotions(list_of_texts):                
    detected_emotions = []
    for i in range(len(list_of_texts)):
    #for emotional_text in emotion_data:
        emotions = model.predict(np.array([(list_of_texts[i])]))
        emotional_list = [emotions[0][0], emotions[0][1], emotions[0][2], emotions[0][3], emotions[0][4], emotions[0][5]]
        detected_emotions.append(emotional_list)
    return emotional_list

# Mean of emotions of tweets in detected emotions
def emotion_mean(detected_emotions):
    mean_emotions = np.array([0, 0, 0, 0, 0, 0])
    for onetweet in detected_emotions:
        mean_emotions = mean_emotions + np.array(onetweet) 

    mean_emotions = mean_emotions / np.array([len(detected_emotions), len(detected_emotions), len(detected_emotions), len(detected_emotions), len(detected_emotions), len(detected_emotions)])
    return mean_emotions


#print(emotion_mean(detected_emotions))
