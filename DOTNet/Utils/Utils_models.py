
#description     :Have functions to get optimizer and loss
#usage           :imported in other files
#python_version  :3.5.4
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
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
from keras import losses
from sklearn.preprocessing import label_binarize


def get_optimizer():
 
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam

def normalize_data(values):
    from math import sqrt

    scaler = StandardScaler()
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)

    return normalized


def lr_decay(epoch):
        return 0.1 * math.pow(0.1, epoch)

def scheduler(epoch):
        if epoch >= 1: #and epoch % 2 == 0:
            plot_generated_images(epoch, model, True,Tmp_ssimlist)
            plot_confusionmatrix(epoch, model)
            plot_roc_curve(model)
            # lr schedule callback
            # lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay)

            PlotLosses()
            if epoch == 5 or (epoch >= 10 and epoch % 10 == 0):
                generator.save('./results/deep_spa_mse_only.h5')


        return K.get_value(model.optimizer.lr)



def calculateDistance(image1, image2):
    distance = 0
    image1 =   np.reshape(image1 ,(128, 128) )
    image2 =   np.reshape(image2 ,(128, 128) )
    i,j = np.unravel_index(image1.argmax(), image1.shape)
    f,h= np.unravel_index(image2.argmax(), image2.shape)
    x = np.array((i ,j))
    y = np.array((f, h))
    dist = np.linalg.norm(np.abs(x-y))
    from scipy.spatial import distance
    dist = distance.euclidean(y,x)
    return dist

def psnr(img1, img2):
    """
    Assuming img2 is the ground truth, we take it's PIXEL_MAX

    :param img1: Synthesized image
    :param img2: Ground Truth
    :return:
    """
    # normalize images between [0, 1]
    epsilon = 0.00001
    img2_n = img1 / (img2.max() + epsilon)
    img1_n = img2 / (img2.max() + epsilon)

    PIXEL_MAX = 1.0

    mse = np.mean((img1_n - img2_n) ** 2)
    if mse == 0:
        return 35
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

def Dice (im1,im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def FuzzyJaccard (im1,im2):
    A = im1.flatten()
    B = im2.flatten()
    minAB= np.minimum(A,B)
    maxAB= np.maximum(A,B)
    union= minAB.sum()/maxAB.sum()
    return union

