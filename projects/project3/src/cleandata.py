import pandas as pd
from pathlib import Path
import json
import csv
from tqdm import tqdm as tqdm
DATA = Path("../data")
BUSINESS = DATA / "business.json"
USER = DATA / "user.json"
REVIEW = DATA / "review.json"
SAVEPATH = DATA / "yelp.csv"

business = {}
columns = ['name', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open']
#categories = Counter()

with open(BUSINESS) as fin:
    for row in tqdm(fin):
        data = json.loads(row)
        vals = {k: v for k, v in data.items() if k in columns}
        if data['attributes'] is not None:
            for k, v in data['attributes'].items():
                if '{' in v:
                    v = eval(v)
                    for k, v_ in v.items():
                        vals[k] = v_
                else:
                    vals[k] = v
        #if data['categories'] is not None:
        #    categories_ = data['categories'].split(', ')
        #    categories.add(categories_)
        #    for category in data['categories'].split(', '):
        #        vals[category] = True
        business[data['business_id']] = vals


columns =['review_count', 'yelping_since', 'useful', 'funny', 'cool', 'friends', 'fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos'] 
users = {}
with open(USER) as fin:
    for row in tqdm(fin):
        data = json.loads(row)
        vals = {k: v for k, v in data.items() if k in columns}
        vals['friends'] = data['friends'].count(',') + 1
        users[data['user_id']] = vals


keys = list({k for _, v in business.items() for k in v})
ukeys = list({k for _, v in users.items() for k in v})
keys = keys + ukeys + ['time', 'rating']

from dateutil import parser
def timesince(date1, date2):
    return (parser.parse(date1) - parser.parse(date2)).total_seconds()


reviews = []
df = pd.DataFrame(columns=keys)
df.to_csv(SAVEPATH)
with open(REVIEW) as fin:
    for row in tqdm(fin):
        data = json.loads(row)
        try:
            b = business[data['business_id']]
            # Must copy the user
            u = {k: v for k, v in users[data['user_id']].items()}
            time = timesince(data['date'], u['yelping_since'])
            u.pop('yelping_since')
            reviews.append({**u, **b, 'time': time, 'rating': data['stars']})
        except KeyError as e:
            continue

        if len(reviews) >= 10_000:
            df = pd.DataFrame(reviews, columns=keys)
            df.to_csv(SAVEPATH, mode='a', header=False)
            reviews = []
            del df
