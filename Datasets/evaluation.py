from fcntl import DN_DELETE
import json
from this import d
import pandas as pd

# read file
read_file = open("Raiders of the Lost Kek/first2500lines.txt", "r")
lines = read_file.readlines()

# EXAMPLE
# counting how often a country appears
countries = dict()

# iterate over lines
# -> each line is a new thread
# -> ndjson file type means each line is a new json object
for line in lines:
    # deserialize json into python dictionary
    data = json.loads(line)
    # disregard all threads with less than 10 replies
    if data["posts"][0]["replies"] < 10: continue
    
    # FOR every post IN the thread, DO...
    for i in range(len(data["posts"])):
        # get country of origin post was posted from
        # -> if not available -> "unknown"
        try:
            country_name = data["posts"][i]["country_name"]
        except KeyError:
            country_name = "unknown"

        # IF country of origin is in dictionary of countries
        # -> increase count by one
        if country_name in countries:
            countries[country_name] += 1
        # ELSE add country to dictionary
        else:
            countries[country_name] = 1

print(countries)


# RESULT
"""
$ time python evaluation.py
{'Canada': 5960, 'United States': 46342, 'Netherlands': 1765, 'Israel': 449, 'Germany': 3501, 'Croatia': 337, 'United Kingdom': 12499, 'Australia': 5713, 'Russian Federation': 388, 'Latvia': 105, 'Greece': 472, 'Sweden': 1146, 'Spain': 714, 'Italy': 571, 'Poland': 814, 'Norway': 796, 'Hungary': 258, 'Slovenia': 315, 'Austria': 1088, 'Bulgaria': 346, 'Ireland': 718, 'Denmark': 388, 'Mexico': 782, 'Finland': 1282, 'Portugal': 342, 'Singapore': 171, 'New Zealand': 607, 'Argentina': 384, 'France': 864, 'Morocco': 18, 'Belgium': 302, 'Montenegro': 63, 'Taiwan': 38, 'Colombia': 160, 'Thailand': 63, 'Czech Republic': 189, 'Brazil': 949, 'Philippines': 439, 'Romania': 508, 'Lithuania': 122, 'Dominican Republic': 22, 'Japan': 423, 'Turkey': 184, 'Puerto Rico': 106, 'Estonia': 117, 'Chile': 146, 'Hong Kong': 39, 'Ukraine': 62, 'Switzerland': 293, 'Georgia': 23, 'Malaysia': 106, 'Monaco': 46, 'Serbia': 218, 'Moldova': 48, 'Peru': 42, 'India': 164, 'South Africa': 153, 'Indonesia': 32, 'Malta': 19, 'Macao': 5, 'Uruguay': 57, 'Costa Rica': 29, 'Kuwait': 2, 'Sri Lanka': 6, 'Panama': 91, 'Algeria': 63, 'Zimbabwe': 60, 'United Arab Emirates': 11, 'Jordan': 27, 'Macedonia': 64, 'Libya': 28, 'Ecuador': 25, 'Trinidad and Tobago': 9, 'Bosnia and Herzegovina': 82, 'Slovakia': 86, 'Iceland': 34, 'Belarus': 41, 'South Korea': 88, 'Vietnam': 13, 'Egypt': 44, 'Falkland Islands': 1, 'Europe': 2, 'Paraguay': 22, 'Pakistan': 6, 'Albania': 48, 'El Salvador': 40, 'Belize': 24, 'Jersey': 7, 'China': 2, 'Cyprus': 30, 'Guadeloupe': 1, 'Isle of Man': 3, 'Tunisia': 17, 'Palestine': 3, 'Luxembourg': 12, 'Kazakhstan': 10, 'British Virgin Islands': 35, 'Sudan': 2, 'Svalbard and Jan Mayen': 4, 'Cura&ccedil;ao': 20, 'Venezuela': 139, 'Faroe Islands': 13, 'Andorra': 3, 'Aruba': 6, 'French Guiana': 10, 'Guernsey': 6, 'Lebanon': 4, 'Bahamas': 5, 'Mauritius': 3, 'Afghanistan': 36, 'Bolivia': 3, 'Dominica': 2, 'Namibia': 3, 'Greenland': 9, 'Nicaragua': 6, 'New Caledonia': 3, 'Cocos (Keeling) Islands': 1, "C&ocirc;te d'Ivoire": 2, 'Honduras': 1, 'Botswana': 7, 'Azerbaijan': 2, 'Niue': 1, 'Jamaica': 1, 'Tuvalu': 4, 'Mongolia': 73, 'Tonga': 1, 'Bangladesh': 1, 'Uzbekistan': 1, 'Iraq': 1, 'Martinique': 3, 'Guatemala': 15, 'unknown': 1, 'Bermuda': 3, 'Tokelau': 1}
"""
# timing results
"""
python evaluation.py  1.51s user 0.24s system 25% cpu 6.856 total
"""