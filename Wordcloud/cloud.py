import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from datasets import load_dataset
import sys, os, re
import numpy as np
from PIL import Image
os.chdir(sys.path[0])

dataset = load_dataset("hugginglearners/russia-ukraine-conflict-articles")
# dataset.shape

"""
# create textfile
with open('data.txt', 'w') as f:
    for i in range(407):
        print(i)
        text = dataset['train'][i]["articles"]
        f.write(text)
        f.write("\n")

text = open('data.txt').read()
"""

# Create Text string with all articles
text = ""
for i in range(407):
    string = dataset['train'][i]["articles"]
    text = text + "\n" + string

# combining words with the same meaning and deleting symbols
text = re.sub(r"http\S+", "", text)
text = re.sub(r"#\S+", "", text)
text = re.sub(r"@\S+", "", text)
text = re.sub(r"’", "", text)
text = re.sub(r"ing", "", text)
text = re.sub(r"last", "", text)
text = re.sub(r"Wednes", "", text)
text = re.sub(r"much", "", text)
text = re.sub(r"likely", "", text)
text = re.sub(r"said", "", text)
text = re.sub(r"say", "", text)
text = re.sub(r"saying", "", text)
text = re.sub(r"Thurs", "", text)
text = re.sub(r"Tues", "", text)
text = re.sub(r"Fri", "", text)
text = re.sub(r"Mon", "", text)
text = re.sub(r"thing", "", text)
text = re.sub(r"including", "", text)
text = re.sub(r"already", "", text)
text = re.sub(r"one", "", text)
text = re.sub(r"two", "", text)
text = re.sub(r"three", "", text)
text = re.sub(r"four", "", text)
text = re.sub(r"first", "", text)
text = re.sub(r"want", "", text)
text = re.sub(r"even", "", text)
text = re.sub(r"day", "", text)
text = re.sub(r"told", "", text)
text = re.sub(r"Mr", "", text)
text = re.sub(r"Thursday", "", text)
text = re.sub(r"official", "", text)
text = re.sub(r"according", "", text)
text = re.sub(r"made", "", text)
text = re.sub(r"Friday", "", text)
text = re.sub(r"still", "", text)
text = re.sub(r"will", "", text)
text = re.sub(r"Russia", "Russian", text)
text = re.sub(r"Russias", "Russian", text)
text = re.sub(r"Russiann", "Russian", text)
text = re.sub(r"Ukraines", "Ukraine", text)
text = re.sub(r"Ukrainian", "Ukraine", text)
text = re.sub(r"Vladimir Putin", "Putin", text)
text = re.sub(r"Volodymyr Zelenskiy", "Zelenskiy", text)
text = re.sub(r"president Volodymyr", "Zelenskiy", text)


# WORDCLOUD
stopwords    = STOPWORDS
mask         = np.array(Image.open('ukraine.png'))
ukrainecolor = ImageColorGenerator(np.array(Image.open('ukraine.png')))

wc = WordCloud(
    mask=mask,
    color_func=ukrainecolor,
    colormap="viridis",
    background_color='black',
    stopwords=stopwords,
    min_word_length=2,
    include_numbers=False,
    height = 1250,
    width = 2000
)

wc.generate(text)

wc.to_file('cloud.png')

