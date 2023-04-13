# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:09:13 2023

@author: adamr
"""
import re
import os

class PipelineConfiguaration:
    def __init__(self, before, after, subreddits, non_mh_subreddits, comments_per_class, data_file_path, output_file_path, min_post_len=50, items_per_class=1250, test_size=0.2, user_grp=False, use_api=False):
        self.before = self.set_date(before)
        self.after = self.set_date(after)
        self.subreddits = self.set_subreddits(subreddits)
        self.non_mh_subreddits = self.set_subreddits(non_mh_subreddits)
        self.comments_per_class = self.set_positive_int(comments_per_class)
        self.data_file_path = self.set_path(data_file_path)
        self.output_file_path  = self.set_path(output_file_path)
        self.min_post_len = self.set_positive_int(min_post_len)
        self.items_per_class = self.set_positive_int(items_per_class)
        self.test_size = self.set_val_bet_0_1(test_size)
        self.use_api = self.set_bool(use_api)
        self.user_grp = self.set_bool(user_grp)
        
    def set_date(self, date):
        #Check that date is an actual date.
        check = 1 if len(re.findall(r"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]", date))>=1 else 0
    
        if check == True:
            return date
        else:
            raise ValueError('DateError: String Provided is not in date format.')
    
    def set_subreddits(self, subreddits):
        if len(subreddits)>1:
            return subreddits
        else:
            raise ValueError('TooSmallNumSubreddits: Need at least two subreddits to be supplied for classification task.')
    
    def set_positive_int(self, num):
        if type(num) == int and num > 0:
            return num
    
    def set_path(self, path):
        if os.path.exists(path):
            return path
        else:
            raise ValueError('FilePathNotFoundError: Path provided does not exist.')  
    
    def set_val_bet_0_1(self, num):
        if num > 0 and num < 1:
            return num
        else:
            raise ValueError('TestSizeError: Test Size must lie between 0 and 1.')
    
    def set_bool(self, val):
        if type(val) == bool:
            return val 
        else:
            raise ValueError('BooleanError: Value Provided is not of type boolean.')
        