# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:27:15 2023

@author: adamr
"""

import praw
import pandas as pd
from datetime import datetime
import numpy as np

class RedditDataAcquirerPraw:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id="Zt74jpr-6BWjILc9spsl8w",
            client_secret="hpeu4u8Ovzz6HGJldtBfoGbmRourqw",
            user_agent="Cyber Legacy",
        )
    
    def get_user_data(self, author):
        submissions = self.reddit.redditor(author).submissions.new(limit=100)
        data = []
        for link in submissions:
            score = link.score
            text = link.selftext
            title = link.title
            subreddit = link.subreddit
            author = author
            date = datetime.utcfromtimestamp(link.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            df = pd.DataFrame({'score':[score], 'selftext': [text], 'title':[title], 'author':[author], 'subreddit':[subreddit], 'date-time':[date]})
            data.append(df)
        return pd.concat(data)
    
data = RedditDataAcquirerPraw()


df = []
users = ['anechointhedark', 'EDPostRequests', 'SafetySnorkel']
for i in range(len(users)):
    submissions = data.get_user_data(users[i])
    df.append(submissions)
df = pd.concat(df)

df['Model Class'] = df['subreddit']
c = pd.DataFrame({'Model Class Confidence':[np.random.normal(0.4, 0.1) for i in range(len(df))]})
df = df.reset_index(drop=True)
df = pd.concat([df, c], axis=1)

df.to_csv(r'Dashboard Data.csv', index=False)

