import tensorflow as tf 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense, Flatten,Dropout, LSTM, SimpleRNN,Input,Conv1D,Conv2D,concatenate,Activation,MaxPooling1D,Masking,GRU,Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tensorflow.keras import backend as K
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Concatenate
import sklearn.metrics as metrics
import pickle
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import LambdaCallback
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
import keras

bestLoss = 1000000000000000000
bestWeights = None

def storeWeights(e, logs):
    global bestLoss
    if logs['val_loss'] < bestLoss:
        bestLoss = logs['val_loss']
        bestWeights = model.get_weights()

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for i in range(physical_devices.shape[0]):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
except:
  #    Invalid device or cannot modify virtual devices once initialized.
     print('pass')
    
gpu_fraction = 0.8
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
  
    



    
print("Defining the Metrics....")
def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
        y_pred = y_pred[:,1:2]
        y_true = y_true[:,1:2]
    return y_true, y_pred

def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def auroc(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
