
#description     :Have functions to get optimizer and loss
#usage           :imported in other files
#python_version  :3.5.4
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.models import model_from_json
import os
import itertools
import numpy as np
import pandas as pd
import glob
import math
from numpy import genfromtxt
from keras import optimizers
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time
import keras as keras
from math import sqrt
from sklearn.utils import shuffle
from sklearn import manifold
from keras import backend as K
from sklearn.metrics import mean_squared_error,median_absolute_error
from keras.models import load_model
import timeit
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix
from  skimage.metrics import structural_similarity as ssim
import skimage
# print(skimage.__version__)
from keras import losses
from sklearn.preprocessing import label_binarize


class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))

def FuzzyJaccard_distance_loss(y_true, y_pred,  n_channels=1):
 	jac = 0
 	for c in range(n_channels):
            true_batch_c = y_true[:, :, :, c]
            pred_batch_c = y_pred[:, :, :, c]
            intersect_batch = (K.minimum(true_batch_c, pred_batch_c))
            union_batch = (K.maximum(true_batch_c, pred_batch_c))
            intersect = K.sum(K.sum(intersect_batch, axis=-1), axis=-1)
            union = K.sum(K.sum(union_batch, axis=-1), axis=-1)
            j = intersect / union
            jac += K.mean(j)
 	union= jac / n_channels
 	union= (-K.log(K.clip(union, K.epsilon(), None) )) #**0.3
 	return union # (1- union )




def loss_r(alpha,batch_size):
    def custom_loss_func_r(y_true, y_pred):
        return custom_loss_r(y_true, y_pred, alpha,batch_size)
    return custom_loss_func_r

def custom_loss_r(y_true, y_pred, alpha,batch_size):  
    loss =losses.mean_squared_error(y_true, y_pred)
    loss+= alpha *FuzzyJaccard_distance_loss(y_true, y_pred)

    return  loss


