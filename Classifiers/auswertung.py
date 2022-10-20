import csv
import matplotlib.pyplot as mt

#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
#plt.plot(days,y)
#plt.gcf().autofmt_xdate()
#plt.show()


with open('emotion_data.csv') as file:
    emotion_data = csv.reader(file)
    sadness = []
    joy = []
    love = []
    anger = []
    fear = []
    suprise = []
    for day in emotion_data:
        sadness.append(float(day[0]))
        joy.append(float(day[1]))
        love.append(float(day[2]))
        anger.append(float(day[3]))
        fear.append(float(day[4]))
        suprise.append(float(day[5]))

days_1 = list(range(len(sadness)))

mt.plot(days_1, sadness, 
        days_1, joy,
        days_1, love,
        days_1, anger,
        days_1, fear,
        days_1, suprise)
mt.legend(["Sadness", "Joy", "Love", "Anger", "Fear", "Suprise"])
mt.xlabel("Date (2022)", 
           family='serif', 
           color='k', 
           weight='normal', 
           size = 16,
           labelpad = 6)
mt.ylabel("Emotion (%)", 
           family='serif', 
           color='k', 
           weight='normal', 
           size = 16,
           labelpad = 6)
#mt.xticks(list(range(0, 230, 10)))
mt.xticks(list(range(0, 240, 10)), ["22/02", "04/03", "14/03", "24/03", "03/04", "13/04", "23/04", "03/05", "13/05", "23/05", "02/06", "12/06", "22/06", "02/07", "12/07", "22/07", "01/08", "11/08", "21/08", "31/08", "10/09", "20/09", "30/09", "10/10"])
mt.grid(True)
mt.show()
