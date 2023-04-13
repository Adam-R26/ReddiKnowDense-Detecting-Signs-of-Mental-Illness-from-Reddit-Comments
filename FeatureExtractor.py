# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:02:46 2023

@author: adamr
"""
from transformers import RobertaTokenizerFast, TFRobertaModel
from sklearn.model_selection import train_test_split

class FeatureExtractor:
    def __init__(self, test_size=0.2):
        self._test_size = test_size
    
    def extract_features(self, X, Y, method='roberta', output_type='pooler_output'):
        #Extract features using required method.
        if method == 'roberta':
            X = [X[i * 32:(i + 1) * 32] for i in range((len(X) + 32 - 1) // 32)]
            print('Num Batches:', len(X))
            count = 1
            encoded = []
            for batch in X:
                batch = self.get_roberta_features(batch)
                encoded = encoded + list(batch[output_type])
                print('Batch ' + str(count)+': Done')
                count+=1
            X = encoded
            
        #Split into train and test: Random state for reproducible results.
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self._test_size, random_state=123, stratify=Y)
        
        return X_train, X_test, Y_train, Y_test
    
    def extract_features_X(self, X, method='roberta', output_type='pooler_output'):
        #Extract features using required method.
        if method == 'roberta':
            X = [X[i * 32:(i + 1) * 32] for i in range((len(X) + 32 - 1) // 32)]
            print('Num Batches:', len(X))
            count = 1
            encoded = []
            for batch in X:
                batch = self.get_roberta_features(batch)
                encoded = encoded + list(batch[output_type])
                print('Batch ' + str(count)+': Done')
                count+=1
            X = encoded
        return X

    def get_roberta_features(self, X):
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        model = TFRobertaModel.from_pretrained('roberta-base')
        encoded_input = tokenizer(X, return_tensors='tf', padding=True, truncation=True)
        output = model(encoded_input)
        return output
    
    