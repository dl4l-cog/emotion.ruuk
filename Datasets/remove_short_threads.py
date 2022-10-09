import os
import json
import pandas as pd

# open files to read from and write to
filepath = "Raiders of the Lost Kek/first2500lines.txt"
filename = os.path.splitext(filepath)[0]
file_ext = os.path.splitext(filepath)[1]

read_file = open(filename + file_ext, "r")
lines = read_file.readlines()

write_file = open(filename + "-cleaned_threads" + file_ext, "a")

# REMOVE THREADS < 10 posts
# iterate over lines
# -> each line is a new thread
# -> ndjson file type means each line is a new json object
for line in lines:
    # deserialize json into python dictionary
    data = json.loads(line)
    # disregard all threads with less than 10 replies
    if data["posts"][0]["replies"] < 10: continue
    write_file.write(line + "\n")

write_file.close()
