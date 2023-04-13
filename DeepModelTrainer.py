# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:49:00 2023

@author: adamr
"""
import tensorflow as tf
from tensorflow import models, layers
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import callbacks
import numpy as np

class DeepModelTrainer:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        print(self.x_train[0:10])
        self.y_train = y_train
        print(self.y_train[0:10])
        
        self.roberta_preprocess = hub.KerasLayer("https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1")
        self.roberta_encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/roberta_en_cased_L-12_H-768_A-12/1")
        self.num_classes = len(np.unique(y_train))
    
    
    def train_all_models(self, epochs=1):
        earlystopping = callbacks.EarlyStopping(monitor ="val_accuracy",
                                        mode ="min", patience = 3,
                                        restore_best_weights = True)
        
        #Train models
        r_cnn = self.configure_roberta_cnn()
        history_r_cnn = r_cnn.fit(self.x_train, self.y_train, validation_split=0.05, callbacks=[earlystopping], epochs=epochs)
        r_mlp = self.configure_roberta_mlp()
        history_r_mlp = r_mlp.fit(self.x_train, self.y_train, validation_batch_size=0.05, callbacks=[earlystopping], epochs=epochs)
        
        #Save trained models and histories
        self.models = {'r_cnn': r_cnn, 'r_mlp': r_mlp}
        self.histories = {'r_cnn': history_r_cnn, 'r_mlp': history_r_mlp}
    
    def train_model(self, model_name, epochs=1):
        earlystopping = callbacks.EarlyStopping(monitor ="val_accuracy",
                                        mode ="min", patience = 3,
                                        restore_best_weights = True)
        
        models = {'r_cnn': self.configure_roberta_cnn(), 'r_mlp': self.configure_roberta_mlp()}
        model = models[model_name]
        model_history = model.fit(self.x_train, self.y_train, validation_split=0.05, callbacks=[earlystopping], epochs=epochs)
        
        return model
    
    def configure_roberta_cnn(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        encoder_inputs = self.roberta_preprocess(text_input)
        outputs = self.roberta_encoder(encoder_inputs)
        #net = outputs['pooled_output'] # [batch_size, 768].
        net = outputs["sequence_output"] # [batch_size, seq_length, 768]
        net = tf.keras.layers.Conv1D(32, (2), activation='relu')(net)
        net = tf.keras.layers.MaxPooling1D(2)(net)
        net = tf.keras.layers.Conv1D(64, (2), activation='relu')(net)
        net = tf.keras.layers.GlobalMaxPool1D()(net)
        net = tf.keras.layers.Dense(512, activation="relu")(net)
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(self.num_classes, activation="softmax", name='classifier')(net)
        model = tf.keras.Model(text_input, net)
        model.compile(optimizer=tf.keras.optimizers.experimental.AdamW(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))
        model.summary()
        return model
    
    def configure_roberta_cnn(self, dropout_value=0.3, num_classes=8):
        model = models.Sequential()
        model.add(layers.Input(shape=()), dtype=tf.string, name='text')
        model.add(layers.Conv1D(32, (2), activation='relu'))
        model.add(layers.Dropout(dropout_value))
        model.add(layers.Conv1D(64, (2), activation='relu'))
        model.add(layers.MaxPooling1D((2)))
        model.add(layers.Dropout(dropout_value))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.dropout(0.1))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        model.summary()
        return model
        
    
    def configure_roberta_mlp(self):            
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.roberta_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.roberta_encoder, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dense(512, activation="relu")(net)
        net = tf.keras.layers.Dropout(0.2)(net)
        net = tf.keras.layers.Dense(self.num_classes, activation="softmax", name='classifier')(net)
        model = tf.keras.Model(text_input, net)
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy())
        model.summary()
        return model
        