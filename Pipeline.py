# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:58:19 2023

@author: adamr
"""
from RedditDataAcquirer import RedditDataAcquirer
from RedditDataPreprocessor import RedditDataPreprocessor
from ModelTrainer import ModelTrainer
from FeatureExtractor import FeatureExtractor
from ModelEvaluator import ModelEvaluator
from SelfTrainer import SelfTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.acquirer = RedditDataAcquirer(config.before, config.after, config.subreddits+config.non_mh_subreddits, config.comments_per_class, config.data_file_path)
        self.preprocessor = RedditDataPreprocessor()
        self.extractor = FeatureExtractor(config.test_size)
        
    def save_training_data(self, X_train, X_test, Y_train, Y_test, encoders, decoders, X_unlabelled):
        data = [X_train, X_test, Y_train, Y_test, encoders, decoders, X_unlabelled] 
        fileNames = ['X_train', 'X_test','Y_train', 'Y_test', 'encoders', 'decoders', 'X_unlabelled']
        for i in range(len(data)):
                fileName = fileNames[i] + '.p'
                with open(fileName, "wb" ) as f:
                    pickle.dump(data[i], f)
                    
    def load_training_data(self):
        fileNames = ['X_train', 'X_test','Y_train', 'Y_test', 'encoders', 'decoders', 'X_unlabelled']
        for i in range(len(fileNames)):
            fileName = fileNames[i] + '.p'
            with open(fileName, 'rb') as f:
                fileNames[i] = pickle.load(f) 
        
        return fileNames
    
    def test_deep_learning_architectures(self, X_train, Y_train, X_test, Y_test, decoders):
        #Train models based on data found. 
        #self, x_train, y_train, best_hyperparameters=None, num_classes=9, dropout_value=0.3, conv_layers=1, dense_neurons=512, dense_layers=3
        dropout_values = [0.5]#, 0.4]#,0.3,0.4]
        conv_layers = [2]#,3,4]
        dense_neurons = [512]#, 256 , 1024] #1024
        dense_layers = [2]#,3,4,5]#,2,3]
        models = {}
        metrics = {}
        
        for p in range(len(dropout_values)):
            for q in range(len(conv_layers)):
                for r in range(len(dense_neurons)):
                    for x in range(len(dense_layers)):
                        trainer = ModelTrainer(X_train, Y_train, num_classes=9, dropout_value=dropout_values[p], conv_layers=conv_layers[q], dense_neurons=dense_neurons[r], dense_layers=dense_layers[x])
                        trainer.train_all_models()
                        models[str(dropout_values[p]), str(conv_layers[q]), str(dense_neurons[r]), str(dense_layers[x])] = trainer.models
        
        #Evaluate performance
        for key in models.keys():
            evaluator = ModelEvaluator(X_test, Y_test, models[key], decoders)
            metric = evaluator.benchmark_models()
            metrics[key] = metric
        
        with open('Metrics Dropout 2000.p', 'wb') as f:
            pickle.dump(metrics, f)
        
    def run(self):
        #Get desired data
        data = self.acquirer.acquire_data(self.config.use_api)
        
        #Apply preprocesing steps
        X, X_unlabelled, Y, encoders, decoders = self.preprocessor.preprocess_data(data, self.config.subreddits, self.config.non_mh_subreddits, self.config.min_post_len, self.config.items_per_class, self.config.user_grp)
    
        #Extract features for Traditional Model Training
        X_train, X_test, Y_train, Y_test = self.extractor.extract_features(list(X), list(Y), 'roberta', 'pooler_output')
        
        X_unlabelled = self.extractor.extract_features_X(list(X_unlabelled))
        
        #Save training data
        self.save_training_data(X_train, X_test, Y_train, Y_test, encoders, decoders, X_unlabelled)
        
        #Load in saved training data.
        X_train, X_test, Y_train, Y_test, encoders, decoders, X_unlabelled = self.load_training_data()
    
        #Train all ML Models on initial training set.
        self.trainer = ModelTrainer(X_train, Y_train)
        self.trainer.train_all_models(5)
        
        #Get Initial Performance
        evaluator = ModelEvaluator(X_test, Y_test, self.trainer.models, decoders)
        init_metrics = pd.DataFrame(evaluator.benchmark_models())
        
        #Perform self-training
        self_trainer = SelfTrainer(X_train, Y_train, X_test, Y_test, X_unlabelled, self.trainer.models, decoders, self.trainer.best_hyperparameters)
        self_metrics = self_trainer.self_train_single_model('CNN') #, training_data
        
        
        
        return init_metrics, self_metrics, encoders, decoders #, r_cnn_self_results

#a = Pipeline()
#x_train, y_train, x_test, y_test = a.run()

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)


#return X_train, X_train, x_test, y_test
        
        
        