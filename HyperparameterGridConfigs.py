import numpy as np

class HyperparameterGridConfigs:
    def get_xgboost_hyperparam_grid(self):
        param_grid = {
        'min_child_weight': [1],   #, 5, 10],
        'gamma': [0.5], #, 1, 1.5, 2, 5],
        'subsample': [0.6],#, 0.8, 1.0],
        'colsample_bytree': [0.6],#, 0.8, 1.0],
        'max_depth': [3],#, 4, 5]
        }
        
        return param_grid

    def get_svm_hyperparam_grid(self):
        kernel = ['linear', 'rbf']
        C = [1]
        param_grid = dict(kernel=kernel, C=C)
        return param_grid
    
    