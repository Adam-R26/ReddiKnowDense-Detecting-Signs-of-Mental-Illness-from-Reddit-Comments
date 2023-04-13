# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:23:39 2023

@author: adamr
"""
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, x_test:np.array, y_test:np.array, models:object, decoders:dict):
        self.models = models
        self.x_test, self.y_test = x_test, y_test
        self.decoders = decoders
        self.classic_model_names = set(['XGBOOST', 'SVC'])
        
    def benchmark_models(self) -> dict:
         model_names = list(self.models.keys())
         results = {}
         for model_name in model_names:
             results[model_name] = self.model_evaluator(model_name)
         return results
     
        
    def model_evaluator(self, model_name:str) -> dict:
        if model_name in self.classic_model_names:
            predictions = self.models[model_name].predict(np.array(self.x_test))
        else:
            predictions = self.models[model_name].predict(np.array(self.x_test))
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
    
    def graph_performance(metric_dict):
        ##TO IMPLEMENT...
        for key in metric_dict:
            pass