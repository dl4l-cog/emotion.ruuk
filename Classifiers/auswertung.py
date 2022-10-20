import ast
import csv
import numpy as np
import matplotlib.pyplot as plt

with open('emotion_data.csv') as file:
    emotion_data = csv.reader(file)
    sadness = []
    joy = []
    love = []
    anger = []
    fear = []
    surprise = []

    var_sadness = []
    var_joy = []
    var_love = []
    var_anger = []
    var_fear = []
    var_surprise = []

    for day in emotion_data:
        emotions = [float(numeric_string) for numeric_string in day[0].strip('][').split()]
        variances = [float(numeric_string) for numeric_string in day[1].strip('][').split()]

        sadness.append(emotions[0])
        joy.append(emotions[1])
        love.append(emotions[2])
        anger.append(emotions[3])
        fear.append(emotions[4])
        surprise.append(emotions[5])

        var_sadness.append(variances[0])
        var_joy.append(variances[1])
        var_love.append(variances[2])
        var_anger.append(variances[3])
        var_fear.append(variances[4])
        var_surprise.append(variances[5])

days_1 = list(range(len(sadness)))

plt.plot(days_1, sadness, 
        days_1, joy,
        days_1, love,
        days_1, anger,
        days_1, fear,
        days_1, surprise)
plt.fill_between(days_1, sadness-1.96*np.square(var_sadness),   sadness+1.96*np.square(var_sadness),   alpha=0.5)
plt.fill_between(days_1, joy-1.96*np.square(var_joy),           joy+1.96*np.square(var_joy),           alpha=0.5)
plt.fill_between(days_1, love-1.96*np.square(var_love),         love+1.96*np.square(var_love),         alpha=0.5)
plt.fill_between(days_1, anger-1.96*np.square(var_anger),       anger+1.96*np.square(var_anger),       alpha=0.5)
plt.fill_between(days_1, fear-1.96*np.square(var_fear),         fear+1.96*np.square(var_fear),         alpha=0.5)
plt.fill_between(days_1, surprise-1.96*np.square(var_surprise), surprise+1.96*np.square(var_surprise), alpha=0.5)
plt.legend(["Sadness", "Joy", "Love", "Anger", "Fear", "Suprise"])

plt.xlabel("Date (2022)", family='serif', color='k', weight='normal', size = 16, labelpad = 6)
plt.ylabel("Emotion (%)", family='serif', color='k', weight='normal', size = 16, labelpad = 6)

#plt.xticks(list(range(0, 230, 10)))
plt.xticks(list(range(0, 240, 10)), ["22/02", "04/03", "14/03", "24/03", "03/04", "13/04", "23/04", "03/05", "13/05", "23/05", "02/06", "12/06", "22/06", "02/07", "12/07", "22/07", "01/08", "11/08", "21/08", "31/08", "10/09", "20/09", "30/09", "10/10"])
plt.grid(True)
plt.show()
