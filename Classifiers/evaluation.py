import numpy as np
import tensorflow as tf
import json
import re
import csv
from dateutil.parser import parse as parse_dt
from datetime import datetime

# Loading the Model
#--------------------------------------------------------------------
model = tf.keras.models.load_model('Classifiers/model700e_1e-4wd.pth')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


####################################################################


# Help functions to make and evalate predictions
#--------------------------------------------------------------------
def text_to_list_of_sentences(text_list):
    text_list = text_list.split('.')
    return text_list

# Mean of emotions of tweets in detected emotions
def emotion_mean(detected_emotions):
    mean_emotions = np.array([0, 0, 0, 0, 0, 0])
    for onetweet in detected_emotions:
        mean_emotions = mean_emotions + np.array(onetweet) 

    mean_emotions = mean_emotions / np.array([len(detected_emotions), len(detected_emotions), len(detected_emotions), len(detected_emotions), len(detected_emotions), len(detected_emotions)])
    return mean_emotions


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
    daily_emotions = []
    for day_of_tweets in tweet_list_by_day:
        predicted_emotions_one_day = predictEmotions(day_of_tweets)
        daily_emotions.append(emotion_mean(predicted_emotions_one_day))
    return daily_emotions


def predictEmotions(list_of_texts):                
    detected_emotions = []
    for i in range(len(list_of_texts)):
    #for emotional_text in emotion_data:
        emotions = model.predict(np.array([(list_of_texts[i])]))
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

#Schaut ob Eingabestring ein reply ist
def isReply(tweet):
    return tweet["in_reply_to_status_id"] is not None

# Schaut ob Eingabestring kürzer ist als drei Worte
def islongerthanthreewords(tweet):
    return len(tweet["full_text"].split()) > 3

# Wandelt json in Liste von Listen um, mit jeweils 5000 Texten (5000 random Tweets von einem Tag)
def json_to_listOfTexts(json_list):        
    tweet_list_by_day = []
    one_day_of_tweets = []
    date = datetime.date(2022, 2, 22)
    for i in range(len(json_list)):
        onetweet = removeFluff(json.loads(json_list[i]))
        # if tweet is not from current date, but newer -> create new sublist
        if parse_dt(onetweet["created_at"]).date() > date:
            tweet_list_by_day.append(one_day_of_tweets)
            one_day_of_tweets = []
            date = parse_dt(onetweet["created_at"]).date()
            print(date) # print date for progress and sanity check
        if onetweet["lang"] == "en" and islongerthanthreewords(onetweet) and not isReply(onetweet): 
            one_day_of_tweets.append(onetweet["full_text"])
    return tweet_list_by_day


######################################################################


# Tweets einlesen
with open('../../Ukraine_Krieg_Tweets.jsonl', 'r') as json_file:
    json_list = list(json_file)

#Tweets einlesen, filtern und predictions machen
tweet_list_by_day = json_to_listOfTexts(json_list)
predictions = wholePrediction(tweet_list_by_day)
print(predictions)
print(len(predictions))

# Ergebnisse in CSV Datei schreiben
file = open('emotion_data.csv', 'w') 
with file:
    writer = csv.writer(file)
    for day in predictions:
        writer.writerow(day)