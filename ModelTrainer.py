# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:36:16 2023

@author: adamr
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from HyperparameterGridConfigs import HyperparameterGridConfigs
from xgboost import XGBClassifier
from sklearn.svm import SVC
from tensorflow.keras import callbacks
from tensorflow.keras import layers, models
import tensorflow as tf

tf.random.set_seed(123)

class ModelTrainer:
    def __init__(self, x_train, y_train, best_hyperparameters=None, num_classes=9, dropout_value=0.3, conv_layers=1, dense_neurons=512, dense_layers=2):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self._hyperparameterConfig = HyperparameterGridConfigs()
        self.classic_model_names = set(['XGBOOST', 'SVC'])
        self.best_hyperparameters = best_hyperparameters
        self.num_classes = num_classes
        self.dropout_value = dropout_value
        self.conv_layers = conv_layers
        self.dense_neurons = dense_neurons
        self.dense_layers = dense_layers
        
    def train_all_models(self, epochs=200):
        print('Tuning Hyperparamters')
        self.tune_model_hyperparamters()

        print('Training Traditional Models')
        #Train traditional models on same train set.
        xgboost = XGBClassifier(random_state=123, n_jobs=7, min_child_weight=self.best_hyperparameters['XGBOOST']['min_child_weight'], 
                                gamma=self.best_hyperparameters['XGBOOST']['gamma'],
                                subsample=self.best_hyperparameters['XGBOOST']['subsample'], 
                                colsample_bytree=self.best_hyperparameters['XGBOOST']['colsample_bytree'],
                                max_depth=self.best_hyperparameters['XGBOOST']['max_depth'])
        xgboost.fit(self.x_train, self.y_train)
        
    
        svc = SVC(random_state=123, C=self.best_hyperparameters['SVC']['C'], kernel=self.best_hyperparameters['SVC']['kernel'], probability=True)
        svc.fit(self.x_train, self.y_train)
        
        #Train DL Models
        earlystopping = callbacks.EarlyStopping(monitor ="val_accuracy",
                                        mode ="auto", patience = 30,
                                        restore_best_weights = True)
        
        #Train models
        cnn = self.configure_roberta_cnn() 
        history_cnn = cnn.fit(self.x_train, self.y_train, validation_split=0.10, callbacks=[earlystopping], epochs=epochs, batch_size=8)
        mlp = self.configure_roberta_mlp()
        history_mlp = mlp.fit(self.x_train, self.y_train, validation_split=0.10, batch_size=32, epochs=1)#, callbacks=[earlystopping], epochs=epochs)
        
        self.models = {'CNN': cnn, 'SVC': svc, 'XGBOOST': xgboost, 'MLP':mlp}
        self.histories = {'CNN': history_cnn, 'MLP': history_mlp}
        
        
    def train_model(self, model_name, epochs=200):
        models = {'CNN': self.configure_roberta_cnn(), 
                  'MLP': self.configure_roberta_mlp(), 
                  'XGBOOST':  XGBClassifier(random_state=123, n_jobs=7, min_child_weight=self.best_hyperparameters['XGBOOST']['min_child_weight'], 
                                          gamma=self.best_hyperparameters['XGBOOST']['gamma'],
                                          subsample=self.best_hyperparameters['XGBOOST']['subsample'], 
                                          colsample_bytree=self.best_hyperparameters['XGBOOST']['colsample_bytree'],
                                          max_depth=self.best_hyperparameters['XGBOOST']['max_depth']),
                  'SVC': SVC(random_state=123, C=self.best_hyperparameters['SVC']['C'], kernel=self.best_hyperparameters['SVC']['kernel'], probability=True)
                 }
        
        earlystopping = callbacks.EarlyStopping(monitor ="val_accuracy",
                                        mode ="min", patience = 30,
                                        restore_best_weights = True)
        
        model = models[model_name]
        if model_name in self.classic_model_names:
            model.fit(self.x_train, self.y_train)
            
        else:
            model.fit(self.x_train, self.y_train, validation_split=0.05, callbacks=[earlystopping], epochs=epochs)
        return model
        
        
    def tune_model_hyperparamters(self):
         svc = self._hyperparamter_tuning(SVC(random_state=123, verbose=True), self._hyperparameterConfig.get_svm_hyperparam_grid(), self.x_train, self.y_train)
         xgboost = self._hyperparamter_tuning(XGBClassifier(random_state=123, n_jobs=7), self._hyperparameterConfig.get_xgboost_hyperparam_grid(), self.x_train, self.y_train)
         self.best_hyperparameters =  {'SVC': svc, 'XGBOOST':xgboost}
        
    def _hyperparamter_tuning(self, model: ClassifierMixin, hyperparameter_grid: dict, x_train: np.array, y_train: np.array) -> dict:
        '''Optimizes hyperparamters for inputted model using repeated stratified k-folds cross validation, returning these optimal hyperparameters'''
        #Set up grid search with 5 fold cross validation.
        cv = StratifiedKFold(n_splits=4, shuffle = True, random_state = 123)
        grid_search = GridSearchCV(estimator=model, param_grid=hyperparameter_grid, n_jobs=7, cv=cv, scoring='f1_macro', error_score=0)

        #Execute grid search
        grid_result = grid_search.fit(x_train, y_train)

        #Summarize the results
        print("Model Training Performance:")
        print('------------------------------------------------------------------')
        print(f"F1: {grid_result.best_score_:3f} using {grid_result.best_params_}")
        print('------------------------------------------------------------------')

        parameters = grid_result.best_params_

        return parameters
    
    def configure_roberta_cnn(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(768,1))) #768, 1
        
        for i in range(self.conv_layers):
            model.add(layers.Conv1D(32, (2), activation='relu'))
            model.add(layers.MaxPooling1D((2)))
        
        for i in range(self.conv_layers):
            model.add(layers.Conv1D(64, (2), activation='relu'))
            model.add(layers.MaxPooling1D((2)))
        
        model.add(layers.Conv1D(128, (2), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dropout(self.dropout_value))
        
        for i in range(self.dense_layers):
            model.add(layers.Dense(self.dense_neurons, activation='relu'))
            
            
        model.add(layers.Dropout(self.dropout_value))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00005), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
        model.summary()
        return model
    #tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001, weight_decay=0.008)
        
    def configure_roberta_mlp(self, dropout_value=0.3):
        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_dim=768))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00005), loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        model.summary()
        return model
