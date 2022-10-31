import os
import json
import collections
import matplotlib.pyplot as plt

# check if file exists and has some content
def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

# open files to read from and write to
filepath = "Raiders of the Lost Kek/pol_062016-112019_labeled-00.ndjson"
filename = os.path.splitext(filepath)[0]
file_ext = os.path.splitext(filepath)[1]

read_file_name = filename + file_ext
read_file = open(read_file_name, "r")
lines = read_file.readlines()

write_file_name   = "reply_num_distribution" + ".txt"
write_file_exists = is_non_zero_file(write_file_name)
write_file = open(write_file_name, "r+") if write_file_exists else open(write_file_name, "w")
# import dictionary if already existing
reply_nums = {int(k):int(v) for k, v in json.load(write_file).items()} if write_file_exists else dict()

# COUNT REPLIES
# iterate over lines
# -> each line is a new thread
# -> ndjson file type means each line is a new json object
for line in lines:
    # deserialize json into python dictionary
    data = json.loads(line)
    try:
        reply_num = data["posts"][0]["replies"]
    except KeyError:
        reply_num = "unknown"

    # IF number of replies is in dictionary
    # -> increase count by one
    if reply_num in reply_nums:
        reply_nums[reply_num] += 1
    # ELSE add number of replies to dictionary
    else:
        reply_nums[reply_num] = 1

sorted_reply_nums = collections.OrderedDict(sorted(reply_nums.items()))
plt.bar(int(sorted_reply_nums.keys()), sorted_reply_nums.values(), color='g')
plt.yscale("log")
plt.show()

json.dump(sorted_reply_nums, write_file)
write_file.close()