import csv
import os
from itertools import count

import pandas
import tweepy
from tweepy.api import API

# input_data_path = '/Users/jaroddeweese/Library/Mobile Documents/com~apple~CloudDocs/Documents/School/CIS531/CIS_531_Projects/term_project/academic_literature/Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior/dataverse_files/hatespeech_chunk_begin_removed.csv'
# input_data_path = 'my_normal_tweets.csv'

def err_for_input():
    raise RuntimeError('Please specify input data path in source code')

def err_for_key():
    raise RuntimeError('Please specify the key via environment var or in source code')

input_data_path = err_for_input()
outfile = 'outfile.csv'

pd_df = pandas.read_csv(input_data_path)
print(pd_df.dtypes)
e = os.environ

auth = tweepy.auth.OAuthHandler(consumer_key=e.get('CONSUMER_KEY', err_for_key()), consumer_secret=e.get('CONSUMER_SECRET', err_for_key()))

client = API(auth_handler=auth)


def get_chunks_of_n(n):
    # If I was doing this again i would have used more_itertools.chunked, but this is already written
    counter = count()
    rows = pd_df.iterrows()
    res = []
    count_val = 0
    while count_val < len(pd_df):
        try:
            count_val = next(counter)
            if count_val > 0 and count_val % n == 0:
                yield res
                counter = count()
                res = []
            else:
                res.append(next(rows))
        except StopIteration:
            return res


n = 10
with open(outfile, 'w') as f_write:
    csv_writer = csv.writer(f_write)
    csv_writer.writerow(['author_id', 'user_handle', 'tweet_id', 'tweet_creation', 'tweet_text', 'in_reply_to', 'follower_count', 'tweet_coding'])
    for row_chunk in get_chunks_of_n(n):
        assert row_chunk
        tweet_ids = [r[1].tweet_id or None for r in row_chunk]
        tweet_id_to_coding_map = {r[1].tweet_id: r[1].maj_label for r in row_chunk}
        print(tweet_ids)
        try:
            tweet_data = client.statuses_lookup(id_=tweet_ids)
        except tweepy.error.RateLimitError:
            import time
            print('sleeping until next rate limit lifts')
            time.sleep(15 * 60 + 10)
            tweet_data = client.statuses_lookup(id_=tweet_ids)

        print(tweet_data)
        for i in tweet_data:
            s = [i.author.id, i.author.screen_name, i.id, i.created_at, i.text, i.in_reply_to_status_id, i.author.followers_count, tweet_id_to_coding_map.get(i.id, -1)]
            csv_writer.writerow(s)
        # break
