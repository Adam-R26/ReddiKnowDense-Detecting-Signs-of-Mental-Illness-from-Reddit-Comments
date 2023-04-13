# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:32:48 2023

@author: adamr
"""
import numpy as np
import pandas as pd
from ModelTrainer import ModelTrainer
from ModelEvaluator import ModelEvaluator
from sklearn.metrics import classification_report

class SelfTrainer:
    def __init__(self, x_train, y_train, x_test, y_test, x_unlabelled, models, decoders, best_hyperparameters, num_classes=9, heuristic=None):
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        
        self.x_unlabelled = x_unlabelled
        #self.x_unlabelled, self.unlabelled_score, self.unlabelled_sentiment = np.array(x_unlabelled[0]), x_unlabelled[1], x_unlabelled[2]
        self.decoders = decoders
        self.models = models
        self.classic_model_names = set(['XGBOOST', 'SVC'])
        self.num_classes = num_classes
        self.best_hyperparameters = best_hyperparameters
        self.heuristic = heuristic
        print('Best Hyperparams 2:' + str(self.best_hyperparameters))
        
        
    def execute_self_training(iterations=3):
        pass
    
    def self_train_single_model(self, model_name, iterations=5, top_percentage=0.05):
        metrics_store = []
        train_store = []
    
        for i in range(iterations):
            df = self.compute_prediction_df(model_name)
            df_add = self.compute_additional_training_data(df.copy(), top_percentage)
            
            #Add records to training set, remove them from unlabelled set of data.
            #return list(df_add.index)
            x_new_records = np.array(self.x_unlabelled)[list(df_add.index)]
            y_new_records = df_add['Class'].to_numpy()
            
            print('X_NEW_RECORDS LEN:', str(len(x_new_records)))
            print('Y_NEW_RECORDS LEN:', str(len(y_new_records)))
            
            #Construct new training sets.   
            self.x_train = np.array(list(self.x_train) + list(x_new_records))
            self.y_train = np.array(list(self.y_train) + list(y_new_records))
            
            #Delete new added data from unlabelled set - Avoiding duplicate additions.
            self.x_unlabelled = np.delete(self.x_unlabelled, list(df_add.index), 0)
            
            #Retrain under new training set
            trainer = ModelTrainer(self.x_train, self.y_train, self.best_hyperparameters)
            model = trainer.train_model(model_name)
            
            #Benchmark Model
            metrics = self.model_evaluator(model, self.x_test, self.y_test, model_name)
            metrics['Model'] = model_name
            metrics['Data Size'] = len(self.x_train) - len(x_new_records)
            
            metrics_store.append(metrics)
            train_store.append((self.x_train, self.y_train))
        
        metrics_store = pd.DataFrame(metrics_store)
        
        
        
        
        return metrics_store, train_store
    
    
    def compute_additional_training_data(self, df, top_percentage, upvote_weight=0.2, sentiment_weight=0.2):   
            if self.heuristic == 'upvotes':
                df['Prob'] = df['Prob']*((upvote_weight*df['score'])+(sentiment_weight*df['sentiment']))
            elif self.heuristic == 'sentiment':
                df['Prob'] = df['Prob']*(sentiment_weight*df['sentiment'])
            elif self.heuristic == 'upvotes and sentiment':
                df['Prob'] = df['Prob']*((upvote_weight*df['score'])+(sentiment_weight*df['sentiment']))
        
            num_records_add_per_class = int((len(df)*top_percentage)/self.num_classes)
            print('NUM PER CLASS:'+ str(num_records_add_per_class))
            df_add = []
            for _class in df.Class.unique():
                tmp_df = df.loc[df['Class']==_class].copy()
                tmp_df = tmp_df.sort_values('Prob', ascending=False)
                tmp_df = tmp_df[0:num_records_add_per_class]
                df_add.append(tmp_df)
                
            df_add = pd.concat(df_add)
            print('LEN DF ADD:' + str(len(df_add)))
            
            return df_add
        
    def compute_prediction_df(self, model_name):
        model = self.models[model_name]
        classic_flag = self.compute_classic_flag(model_name)
    
        #Get predictions for the unlabelled data
        if classic_flag:
            predictions_class = model.predict(self.x_unlabelled)
            predictions_prob = model.predict_proba(self.x_unlabelled)
            predictions_prob = [np.max(i) for i in predictions_prob]
            
        else:
            predictions_class, predictions_prob = self.predict_keras(model, np.array(self.x_unlabelled))
            
        #Merge predictions with original data points in dataframe
        #df = pd.DataFrame({'Class':predictions_class, 'Prob': predictions_prob, 'Score': self.unlabelled_score, 'Sentiment': self.unlabelled_sentiment})
        df = pd.DataFrame({'Class':predictions_class, 'Prob': predictions_prob})
        print('DF:',str(len(df)))
        
        return df
    
    def compute_classic_flag(self, model_name):
        classic_flag=False
        if model_name in self.classic_model_names:
            classic_flag = True
        return classic_flag
        
            
    
    def predict_keras(self, model, x):
        x = np.asarray(x).astype(np.float32)
        predictions = model.predict(x)
        predictions_class = [np.argmax(i)for i in predictions]
        predictions_prob = [np.max(i) for i in predictions]
        return predictions_class, predictions_prob
            
    
    def model_evaluator(self, model:str, x_test, y_test, model_name) -> dict:
        if model_name in self.classic_model_names:
            predictions = model.predict(np.array(x_test))
        else:
            predictions = model.predict(np.array(x_test))
            predictions = [np.argmax(i) for i in predictions]
            
        metrics = classification_report(np.array(self.y_test), np.array(predictions), output_dict=True)
        
        #Get list of all classes in the metric dict
        encoded_classes = list(metrics.keys())
        encoded_classes.remove('accuracy')
        encoded_classes.remove('macro avg')
        encoded_classes.remove('weighted avg')

        #Iterate through encoded classes restoring class names.
        for key in encoded_classes:
            metrics[self.decoders[int(key)]] = metrics.pop(key)

        return metrics
            
            
            
            
            
                
            
            
        
        
        
        
        