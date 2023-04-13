# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:42:32 2023

@author: adamr
"""
from PipelineConfiguration import PipelineConfiguaration
from Pipeline import Pipeline

before = r'2023-03-20'
after  = r'2015-01-01'

subreddits = ['AnorexiaNervosa',
'Anxiety',
'EatingDisorders',
'addiction',
'bipolar',
'bulimia',
'depression',
'ptsd',
'schizophrenia',
'selfharm']

non_mh_subreddits = ['music',
               'travel',
               'Pets',
               'politics',
               'english',
               'datasets',
               'mathematics',
               'science'
              ]

comments_per_class = 10000
data_file_path = r'C:\Users\adamr\Documents\UniversityWork\Y4\COMP592 Individual Project\Code\Input\overall_dataset.csv'
output_file_path = r'C:\Users\adamr\Documents\UniversityWork\Y4\COMP592 Individual Project\Code\Output'
min_post_len = 50
items_per_class = 1250#3125
num_classes = 9
use_api = False
user_grp = True
                                #self, before, after, subreddits, non_mh_subreddits, comments_per_class, data_file_path, output_file_path, min_post_len=50, items_per_class=1250, test_size=0.2, use_api=False
config = PipelineConfiguaration(before, after, subreddits, non_mh_subreddits, comments_per_class, data_file_path, output_file_path, min_post_len, items_per_class, user_grp=user_grp)

def main():
    pipeline = Pipeline(config)
    a = pipeline.run()
    return a
    
b = main()
