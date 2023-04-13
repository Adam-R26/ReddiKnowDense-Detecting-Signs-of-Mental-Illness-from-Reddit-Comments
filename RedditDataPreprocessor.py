# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:08:18 2023

@author: adamr
"""
import re
import string
import pandas as pd
import numpy as np
import pandas as pd 
from transformers import pipeline
from textblob import TextBlob

class RedditDataPreprocessor:
    def preprocess_data(self, data: pd.DataFrame, subreddits, non_mh_subreddits, min_post_len=50, items_per_class=1250, user_grp=False):
        #Generate training examples: Combining Title with Body Text
        data = self.build_attributes(data, subreddits, non_mh_subreddits)
        
        #Remove any deleted submissions
        data = data[(data['author']!='[deleted]') & (data['selftext']!='[deleted]') & (data['author']!='[removed]') & (data['selftext']!='[removed]')].copy()
        
        #Restrict to only self-reported users
        mental, non_mental = self.split_classes(data, subreddits)
        mental = self.restrict_to_self_reported(mental)
        data = pd.concat([mental, non_mental])
        
        
        if user_grp:
            print('entered!')
            data['post_content'] = data['post_content'].astype(str)
            data = data.groupby('author')[['post_content', 'score', 'class']].agg({'post_content':' '.join, 'score': 'mean', 'class': lambda x: x.mode().iat[0]}).reset_index()

        #Clean Data: Case Norm, Punctuation Removal, Email and URL Removal, HTML Character Normalization (Special Character Removal)
        data['post_content'] = data['post_content'].apply(lambda x: self.clean_string(x))
        
        #Remove short submissions
        data = self.remove_short_submissions(data, min_post_len)
        print('Done')


        # #Get sentiment for all data points
        # data['sentiment'] = data['post_content']
        # print('Done')

        # #Scale scores using sentiment
        # data['score_scaled'] = data['score']*data['sentiment']
        
        # data.to_csv(r'Preprocessed User Level Data', index=False)


        #Prune data by score: Get most upvoted comments
        data, left_over_data = self.prune_data_by_score(data, items_per_class)
        
        #Correct Spelling Mistakes
        data['post_content'] = data['post_content'].apply(lambda x: str(TextBlob(x).correct()))
        
        #Get sentiment for unlabelled data.
        x_unlabelled = left_over_data['post_content'].to_numpy()
        print('Done')
        
        #Split into training data and labels
        x = data['post_content'].to_numpy()
        y = data['class'].to_numpy()
        
        #Apply label encoding to target
        encoders = {k:v for v, k in enumerate(np.unique(y).tolist())}
        decoders = {v:k for k, v in encoders.items()}
        y = [encoders[i] for i in y]
        
        return x, x_unlabelled, y, encoders, decoders
    
    def split_classes(self, data, subreddits):
        mental = data.loc[data['subreddit'].isin(subreddits)].copy()
        non_mental = data.loc[~(data['subreddit'].isin(subreddits))].copy()
        return mental, non_mental
         
    def build_attributes(self, data, subreddits, non_mh_subreddits):
        #Generate training examples: Combining Title with Body Text
        data['post_content'] = data['title'] + ' ' + data['selftext']
        
        #Remove non subreddit related posts
        data = data[data['subreddit'].isin(subreddits+non_mh_subreddits)].copy()
        
        #Derive class labels
        print('CLASSES:', sorted(list(data['subreddit'].unique())))
        data['class'] = data['subreddit'].apply(lambda x: self.assign_class(x))
        
        #Drop any duplicate posts
        data = data.drop_duplicates()
        
        return data
        

    def assign_class(self, x):
        x = x.lower()
        if x == 'anorexianervosa':
            return 'eating disorder'
        elif x == 'eatingdisorders':
            return 'eating disorder'
        elif x == 'bulimia':
            return 'eating disorder'
        elif x in ['music', 'travel', 'india','politics','datasets','mathematics','science', 'pets']:
            return 'nonMentalHealth'
        else:
            return x
        
    def restrict_to_self_reported(self, data):        
        user_grp = data.groupby('author_fullname').agg(['sum']).reset_index()
        user_grp = user_grp.droplevel(1, axis=1)
        user_grp['title'] = user_grp['title'].astype(str)
        user_grp['selftext'] = user_grp['selftext'].astype(str)
        user_grp["post_content"] = user_grp["post_content"].apply(lambda x: str(x).lower())
        user_grp["matches"] = user_grp["post_content"].str.split().apply(set(RedditDataPreprocessor.key_words).intersection)
        user_grp['found'] = user_grp["matches"].apply(lambda x: 1 if len(x)!=0 else 0)
        users = user_grp['author_fullname'].tolist()
        data = data.loc[data['author_fullname'].isin(users)].copy()
        
        return data
        
    
    def clean_string(self, text: str) -> str:
        '''Function pre-processes text into a form that can go into the pre-trained models.'''        
        text = str(text).lower()
        
        #Remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        
        #Remove words with numbers in them
        text = re.sub('\w*\d\w*', '', text)
        
        #Remove email addresses'
        text = re.sub('\w*@\w*', '', text)
        
        #Remove numbers
        text = re.sub('\d', '', text)
        
        #Remove special characters
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        
        #Remove all urls
        text = re.sub(r'http\S+', '', text)
        
        
        return text
    
    def prune_data_by_score(self, df: pd.DataFrame, records_per_class):
        classes = df['class'].unique().tolist()
        dfs = []
        dfs_remaining = []
        for _class in classes:
            class_records = df[df['class']==_class].copy()
            class_records = class_records.sort_values('score', ascending=False)
            class_records_dataset = class_records[0:records_per_class].copy()
            other_class_records = class_records[records_per_class+1: records_per_class+1250].copy()#len(class_records)
            dfs_remaining.append(other_class_records)
            dfs.append(class_records_dataset)
            print(str(_class), len(other_class_records))
        
        dataset = pd.concat(dfs)
        dfs_remaining = pd.concat(dfs_remaining)
        
        return dataset, dfs_remaining
    
    def remove_short_submissions(self, df, num_chars):
        df['post_length'] = df['post_content'].apply(lambda x: len(str(x)))
        df = df[df['post_length']>=num_chars].copy()
        return df
    
    def get_sentiment(self, text):
        sentiment_pipeline = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment", truncation=True, padding=True, max_length=50)
        sentiment = sentiment_pipeline(text)[0]
        print(sentiment)
        return sentiment
        
    
    key_words = ['anxiety', 'abixa', 'acp103', 'acp-103', 'acp103', 'akatinol', 'amazeo', 'amidone', 'amipride', 
                 'amisulpride', 'amival', 'aricept', 'aripiprazole', 'ativan', 'atosil', 'avanza', 'avomine', 
                 'axit', 'axura', 'brexpiprazole', 'brintellix', 'buprenex', 'buprenorphine', 'butrans', 
                 'cariprazine', 'celexa', 'cibaliths', 'cipramil', 'citalopram', 'clopine', 'clozapine', 
                 'clozaril', 'convulex', 'dementia', 'depress', 'diastat', 'diastatacudial', 'diazepam', 
                 'dolophine', 'donepezil', 'ebixa', 'effexor', 'epilim', 'episenta', 'epival', 'eskalith', 
                 'exelon', 'fargan', 'farganesse', 'fazaclo', 'fluoxetine', 'galantamine', 'haldol', 'haloperidol', 
                 'heptadon', 'imovane', 'invega', 'lanzek', 'latuda', 'lergigan', 'lithobid', 'lorazepam', 'lurasidone', 
                 'lustral', 'lycoremine', 'memantine', 'memox', 'methadone', 'methadose', 'mirtaz', 'mirtazapine', 'mirtazon',
                 'namenda', 'nivalin', 'nuplazid', 'obsessivecompulsive', 'ocd', 'ocp34712', 'olanzapine', 'opc34712', 'opc-34712',
                 'paliperidone', 'phenergan', 'physeptone', 'pimavanserin', 'promethazine', 'promethegan', 'prothiazine', 'prozac',
                 'quetiapine', 'razadyne', 'receptozine', 'remeron', 'reminyl', 'rgh188', 'rgh-188', 'rgh188', 'risperdal', 'risperidone',
                 'rivastigmine', 'romergan', 'sarafem', 'schizoaffective', 'schizoaffective', 'schizophren', 'selfharm', 'selharm', 
                 'seroquel', 'sertraline', 'solian', 'soltus', 'sominex', 'subutex', 'suicid', 'sulpitac', 'sulprix', 'symoron', 
                 'trintellix', 'valium', 'valproate', 'venlafaxine', 'versacloz', 'vortioxetine', 'zaponex', 'zetran', 'zimovane', 
                 'zispin', 'zoloft', 'zopiclone', 'zypadhera', 'zyprexa', 'bulimia', 'binge', 'anorexia', 'arfid', 'diagnosed with', 
                 'diagnosed history', 'diagnosed mesuffering with', 'history of', 'i was diagnosed', 'i have been diagnosed', 'self-harming'
                 ,'self harming', 'paranoid', 'clinically']
