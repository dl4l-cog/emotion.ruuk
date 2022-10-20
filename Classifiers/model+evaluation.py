#!/usr/bin/env python3
import csv
import json
import re
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from dateutil.parser import parse as parse_dt
from keras import backend as K


# MAGIC NUMBERS
#--------------------------------------------------------------------
EPOCHS = 10
VOCAB_SIZE = 4000


#####################################################################


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
# model = tf.keras.models.load_model('model')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc', f1_m, precision_m, recall_m]
)

model.summary()

# Training the model
model.fit(train_dataset, epochs=EPOCHS,
          validation_data=test_dataset, validation_steps=30)

# Saving the model so it doesn't have to be trained every time
model.save('model')

# print evaluation results
#--------------------------------------------------------------------
loss, accuracy, f1_score, precision, recall = model.evaluate(test_dataset)
print('\n')
print(f'Test Loss:\t{loss:.3f}')
print(f'Test Acc:\t{accuracy:.3f}')
print(f'Test F1:\t{f1_score:.3f}')
print(f'precision / recall:\t{precision:.3f} / {recall:.3f}')


####################################################################


# Help functions to make and evalate predictions
#--------------------------------------------------------------------
def text_to_list_of_sentences(text_list):
    text_list = text_list.split('.')
    return text_list

# calculate runnning mean and variance of emotions of tweets in detected_emotions
def emotion_mean_variance(detected_emotions):
    # intialize running mean and variance
    mean_emotions = np.zeros(6)
    var_emotions  = np.zeros(6)
    counter = 0

    # calculate runnning mean and variance
    # source: https://www.johndcook.com/blog/standard_deviation/
    for tweet in detected_emotions:
        emotions = np.array(tweet)
        mean_emotions_prev = mean_emotions

        if counter > 1:
            mean_emotions += (emotions - mean_emotions) / counter
            var_emotions  += np.multiply(emotions - mean_emotions_prev, emotions - mean_emotions)
            #print(f'Current variances {var_emotions}\r', end="")
        else:
            mean_emotions = emotions

        counter += 1

    return mean_emotions, (var_emotions / (counter - 1))


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


def wholePrediction(tweet_list_by_day):
    daily_emotions_means = []
    daily_emotions_vars  = []

    for day_of_tweets in tweet_list_by_day:
        # predict emotions
        predicted_emotions_one_day = predictEmotions(day_of_tweets)
        # calculate means and variances
        means, variances = emotion_mean_variance(predicted_emotions_one_day)

        daily_emotions_means.append(means)
        daily_emotions_vars.append(variances)

    return daily_emotions_means, daily_emotions_vars


def predictEmotions(list_of_texts):                
    detected_emotions = []
    for i in range(len(list_of_texts)):
    # for emotional_text in emotion_data:
        emotions = model.predict(np.array([(list_of_texts[i])]), verbose=0)
        
        print(f'predicting tweet # {i:4.0f} of {len(list_of_texts):4.0f}\r', end="")

        emotional_list = [emotions[0][0], emotions[0][1], emotions[0][2], emotions[0][3], emotions[0][4], emotions[0][5]]
        detected_emotions.append(emotional_list)

    return detected_emotions


#####################################################################


# Helper functions to read and filter the right tweet text
#--------------------------------------------------------------------

# entfernt Links, Hashtags, Mentions aus Eingabestring
def removeFluff(tweet):
    tweet["full_text"] = re.sub(r" http\S+", "", tweet["full_text"]) #remove URL
    tweet["full_text"] = re.sub(r" #\S+", "", tweet["full_text"])    #remove #
    tweet["full_text"] = re.sub(r" @\S+", "", tweet["full_text"])    #remove @
    return tweet

# schaut ob Tweet eine Antwort auf einen anderen Tweet ist
def isReply(tweet):
    return tweet["in_reply_to_status_id"] is not None

# Schaut ob Eingabestring kürzer als drei Worte ist
def islongerthanthreewords(tweet):
    return len(tweet["full_text"].split()) > 3

# Wandelt json in Liste von Listen um, mit jeweils 5000 Texten (5000 random Tweets von einem Tag)
def json_to_listOfTexts(json_list):        
    tweet_list_by_day = []
    one_day_of_tweets = []
    date = parse_dt("Feb 22 2022").date()
    print(f"starting from {date}")

    for i in range(len(json_list)):
        try:
            onetweet = removeFluff(json.loads(json_list[i]))
        except:
            print(i, json_list[i])
        print('Current date %s\r' % (date.strftime("%Y-%m-%d")), end="") # print date for progress and sanity check

        # if tweet is not from current date, but newer -> create new sublist
        if parse_dt(onetweet["created_at"]).date() > date:
            tweet_list_by_day.append(one_day_of_tweets)
            one_day_of_tweets = []
            date = parse_dt(onetweet["created_at"]).date()
        
        if onetweet["lang"] == "en" and islongerthanthreewords(onetweet) and not isReply(onetweet):
            one_day_of_tweets.append(onetweet["full_text"])

    print('\n')
    return tweet_list_by_day


######################################################################


# Tweets einlesen
with open('../../Ukraine_Krieg_Tweets.jsonl', 'r') as json_file:
    json_list = json_file.read().splitlines()

#Tweets einlesen, filtern und predictions machen
tweet_list_by_day = json_to_listOfTexts(json_list)
prediction_means, prediction_vars = wholePrediction(tweet_list_by_day)
print(prediction_means)
print(len(prediction_means))

# Ergebnisse in CSV Datei schreiben
file = open('emotion_data.csv', 'w') 
with file:
    writer = csv.writer(file)
    for (means, variances) in zip(prediction_means, prediction_vars):
        writer.writerow([means, variances])