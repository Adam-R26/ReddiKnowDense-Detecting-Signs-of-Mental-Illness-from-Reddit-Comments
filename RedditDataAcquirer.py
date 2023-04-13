# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:29:20 2023

@author: adamr
"""

import pandas as pd
import datetime as dt
from pmaw import PushshiftAPI


class RedditDataAcquirer:
    def __init__(self, before_date: str, after_date: str, subreddits: list, number_of_posts: int, data_file_path: str):    
        before_date_split = before_date.split("-")
        after_date_split = after_date.split("-")
        self._before_date = int(dt.datetime(int(before_date_split[0]), int(before_date_split[1]), int(before_date_split[2]) ,0 ,0).timestamp())
        self._after_date = int(dt.datetime(int(after_date_split[0]), int(after_date_split[1]), int(after_date_split[2]) ,0 ,0).timestamp())
        self._subreddits = subreddits
        self._number_of_posts = number_of_posts
        self._data_file_path = data_file_path
        

    def acquire_data(self, use_api_flag: bool) -> pd.DataFrame():
        '''Acquires the data using the API or from memory depending on the "use_api_flag'''
        if use_api_flag and int(self._number_of_posts)>0:
            subreddit_data = []
            for i in range(len(self._subreddits)):
                df_tmp = self._get_subreddit_posts_pmaw(self._subreddits[i])
                df_tmp['Target'] = i
                subreddit_data.append(df_tmp)
            
            df = pd.concat(subreddit_data)
            
        elif use_api_flag and self._number_of_comments<=0:
            raise ValueError("Number of Comments Per Class Parameter Must be More than 0.")
            
        else:
            df = pd.read_csv(self._data_file_path)
            
        return df
    
    def _get_subreddit_posts_pmaw(self, subreddit: str, author=None) -> pd.DataFrame():
        '''Retrieves data from subreddit and returns in dataframe'''
        api = PushshiftAPI()
        if author == None:
            submissions = api.search_submissions(subreddit=subreddit, until=self._before_date, since=self._after_date, limit=self._number_of_posts)
        else:
            submissions = api.search_submissions(subreddit=subreddit, until=self._before_date, since=self._after_date, limit=self._number_of_posts, author=author)
        
        sub_df = pd.DataFrame(submissions)
        return sub_df
    

subreddits = ['depression', 
              'Anxiety',
              'bipolar',
              'schizophrenia',
              'ptsd', 
              'addiction', 
              'selfharm', 
              'AnorexiaNervosa', 
              'EatingDisorders']
   
subreddits = ['music',
              'travel',
              'Pets',
              'politics',
              'english',
              'datasets',
              'mathematics',
              'science'
              ]


# i = 0
# while i<len(subreddits):
#     subreddits_tmp = subreddits[i:i+2]
#     print(subreddits_tmp)
#     acquirer = RedditDataAcquirer('2023-03-20', '2015-01-01', subreddits_tmp, 1000, '')
#     df = acquirer.acquire_data(True)
#     df.to_csv(r'dataset_p_non'+str(i)+'.csv', index=False)
#     i+=2
# acquirer = RedditDataAcquirer('2023-03-20', '2015-01-01', subreddits, 1000, '')
# df = acquirer.acquire_data(True)


# df.to_csv(r'dataset_non_mental_health.csv', index=False)
acquirer = RedditDataAcquirer('2023-03-01', '2023-01-01', ['EatingDisorders'], 1000, '')
df = acquirer._get_subreddit_posts_pmaw('EatingDisorders', 'EDPostRequests')
df.to_csv(r'dataset__1'+'.csv', index=False)

    
    