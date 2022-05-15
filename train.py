from dataread import all_leads_process
from utility import storeWeights, check_units, precision, auroc, f1
import tensorflow as tf 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import ast
import json
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense, Flatten,Dropout, LSTM, BatchNormalization,SimpleRNN,Input,Conv1D,Conv2D,Concatenate,Activation,MaxPooling1D,Masking,GRU,Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score

import sklearn.metrics as metrics

import tensorflow_addons as tfa
from tensorflow.keras.callbacks import LambdaCallback

from sklearn import preprocessing
import keras

gpu_fraction = 1
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


from keras import backend as K

def _bn_relu(layer, params,dropout=0):

    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        params,
        subsample_length=1):

    
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        
        kernel_initializer="he_normal")(layer)
    return layer


def add_conv_layers(layer,num_layers, params):
    for subsample_length in num_layers:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    params,
                    subsample_length=subsample_length)
        layer = _bn_relu(layer, params)
    return layer



gpu_fraction = 1
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def buildModel(time_steps=1691, n_features=157):


    """The branch dealing with the EHR Data"""
    demo_in =Input(shape=(n_features,), name='Demo 1') 
    dense1 = Dense(n_features, activation='relu')(demo_in)
    dense2 = Dropout(0.25)(dense1)

    dense2 = Dense(128, activation='relu')(dense2)
    dense2 = Dropout(0.25)(dense2)    


    """The branch dealing with ECG Data"""
    params = json.load(open('config.json', 'r'))
    inputs = Input(shape=(1691,6),
                   dtype='float32',
                   name='inputs')
    
    layer = add_conv_layers(inputs, params['conv_subsample_lengths'],params)
    layer=  BatchNormalization()(layer)
    layer = add_conv_layers(layer, params['conv_subsample_lengths'],params)
    layer=  BatchNormalization()(layer)

    
    layer = layers.MaxPooling1D(pool_size=4)(layer)
    layer = layers.MaxPooling1D(pool_size=4)(layer)
    layer = layers.MaxPooling1D(pool_size=4)(layer)


    layer = Flatten()(layer)
    d2 = Dense(256,activation = 'relu')(layer)
    d2 = Dropout(0.25)(d2)
    d2=  BatchNormalization()(d2)
    d2 = Dense(16,activation = 'relu')(d2)
    d2 = Dropout(0.25)(d2)
    """Combining both the branches"""
    
    
    merge = Concatenate()(
        [dense2,d2]
    )
    
    
    hidden = Dense(128, activation='relu',name='dense1')(merge)

    
    drop1 = Dropout(0.25)(hidden)

    hidden2 = Dense(1, activation='sigmoid',name='output2')(drop1)
    
    
    #Building the Model
    model = tf.keras.Model(
        inputs = [demo_in, inputs],             
        outputs=hidden2)
    print("Compiling the Model with optimizers and Metrics")
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(curve= 'ROC')
    ]

    model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits= False),metrics = metrics,optimizer = opt)

    return model 

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def trainModel(data):
    # Storing the outcomes 
    outcomes = [ 'Time from PCI to Stroke_1yr', 'Time from PCI to CHF Hospitalization_1yr', 
                'Mortality 1 yr\nYes:1\nNo:0']
    true_variables= []
    pred_variables =[]
    models=[]

    """
       true_variables: an array containing the labels 
       pred_variables: an array containing the predicted results for the outcomes 
       models: A list of the models trained. """           
 
    
    
    """If using the 6 months data as label, remove the comment for the code below"""
    
 #   outcomes2 = ['Time from PCI to Stroke_6mo','Time from PCI to CHF Hospitalization_6mo','Mortality 6 months\nYes:1\nNo:0']
    for outcome in outcomes:
        
        print("Generating results for: ", outcome)
        """ If using the 6 months labels remove the comments from the lines below """
        
        
 #       if outcome == 'Time from PCI to Stroke_1yr':
 #           outcome2 = 'Time from PCI to Stroke_6mo'
 #       elif outcome == 'Time from PCI to CHF Hospitalization_1yr':
 #           outcome2 = 'Time from PCI to CHF Hospitalization_6mo'
 #       elif outcome == 'Mortality 1 yr\nYes:1\nNo:0':
 #           outcome2 = 'Mortality 6 months\nYes:1\nNo:0'
        
        
        ind_var, demo, labels,image_name = all_leads_process(data, outcome)
        
        

        
        
        
        """Readint the Train and Test Images and storing the corresponding labels and information into arrays"""


        
            

        with open('image_train1yr.txt') as f:
            
            image_train_info = f.read()

        image_train_info = ast.literal_eval(image_train_info)

        with open('image_test1yr.txt') as f:
            image_test_info = f.read()

        image_test_info = ast.literal_eval(image_test_info)    
        
        
        """"Code for storing the labels corresponding to the outcomes"""
        
        
        
        

        my_file = open(image_train_info[outcome], "r")

        # reading the file
        image_data = my_file.read()

        image_name_train = image_data.split("\n")


        my_file = open(image_test_info[outcome], "r")

        # reading the file
        image_data = my_file.read()

        image_name_test = image_data.split("\n")
        
        
        for img in range(len(image_name_train)):
            if image_name_train[img][-4:] == '.png' or image_name_train[img][-4:] == '.PNG':
                image_name_train[img] = image_name_train[img][:-4]

        for img in range(len(image_name_test)):
            if image_name_test[img][-4:] == '.png' or image_name_test[img][-4:] == '.PNG':
                image_name_test[img] = image_name_test[img][:-4]
        
        ecg_train =[]
        ecg_test =[]
        
        tab_train =[]
        tab_test = []
        
        labels_train=[]
        labels_test = []
        
        
        """ecg_train/test: array containing ECG data
           tab_train/test: array containing EHR data
           labels_train/test: array containing the corresponding labels"""
        
        for image in image_name_train:
            if image in image_name:
                
                ecg_train.append(ind_var[image_name.index(image)])

                tab_train.append(demo[image_name.index(image)])
                
                labels_train.append(labels[image_name.index(image)])
            
        
        for image in image_name_test:
            if image in image_name:
                
                ecg_test.append(ind_var[image_name.index(image)])

                tab_test.append(demo[image_name.index(image)])
                
                labels_test.append(labels[image_name.index(image)])  

        
        tab_train = np.array(tab_train)
        tab_test= np.array(tab_test)
        
        
        ecg_train = np.array(ecg_train)
        ecg_test = np.array(ecg_test)
        
        
        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)
        print(labels_train)


        

        
        
        model = buildModel(time_steps = 1691, n_features=157)   
        
        
        """Taking the class_weights since its imbalanced dataset"""
        
        
        class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(labels_train),
                                                     labels_train)
        
                                                     
                                                     
        """Defining the callbacks"""
        callback_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        tf.config.run_functions_eagerly(True)
        
        
    
        """Training the model"""

        weight = {i : class_weights[i] for i in range(2)}
       

        history = model.fit([tab_train,ecg_train], labels_train, batch_size=64, validation_split = 0.2, epochs = 100,
                            callbacks=[callback_stopping, callback_scheduler],shuffle=True,class_weight=weight) 
    
        #Storing the actual and predicted labels for the outcomes 
        
        
        preds = model.predict([tab_test,ecg_test])
        
        """Storing the labels"""
        
        true_variables.append(labels_test)
        pred_variables.append(preds)
        
        
        """Storing the model"""
        
        models.append(model)
        


        

    return true_variables, pred_variables, outcomes, models
    

