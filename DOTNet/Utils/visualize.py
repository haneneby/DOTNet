
#description     :Have functions to get optimizer and loss
#usage           :imported in other files
#python_version  :3.5.4
import tensorflow as tf
from Utils.Utils_models import *
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
from keras import losses
from sklearn.preprocessing import label_binarize

global Tmp_ssimlist
Tmp_ssimlist = 0

def plot_generated_images(dir,generated_image,x_train,val =True, examples=15, dim=(1, 2), figsize=(10, 5)):
    fg_color = 'black'
    bg_color =  'white'
    DistanceROI = []
    mselist=[]
    psnrlist=[]
    ssimlist=[]
    Dicelist= []
    FJaccard=[]
    vmin=0
    vmax=25
    scale = np.array(100/25)  
    # PD_label=[]
    # GT_label=[]
    global Tmp_ssimlist
    dirfile= dir+ '/test_generated_image_'
    if val :
        r= examples
    else: 
        r= len(x_train)
    for index in range(r):
            ## plot GT
            fig=plt.figure(figsize=figsize)
            ax1=plt.subplot(dim[0], dim[1], 1)
            ax1.set_title('GT', color=fg_color)
            imgn = np.flipud(x_train[index])/scale 
            im1 = ax1.imshow(imgn.reshape(128, 128))  
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax1.axis('off')
            fig.colorbar(im1, cax=cax, orientation='vertical')

            ## plot Rec1
            ax2=plt.subplot(dim[0], dim[1], 2)
            imgnr = np.flipud(generated_image[index])/scale 
            ax2.set_title('Recons_f1', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax2.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            plt.tight_layout()
            plt.savefig(dirfile+ '-' +str(index)+'.png' )
            
            ## compute metrics
            v=calculateDistance (generated_image[index],x_train[index])#
            DistanceROI.append(v)
            p=psnr(generated_image[index],x_train[index])
            psnrlist.append(p)
            ss_im = ssim(x_train[index].reshape(128, 128), generated_image[index].reshape(128, 128))
            ssimlist.append(ss_im)
            fjacc= FuzzyJaccard(x_train[index],generated_image[index])
            FJaccard.append(fjacc)
            plt.close("all")
 
    FJ_mean= np.mean(FJaccard)
    FJ_std= np.std(FJaccard)
    DistanceROI_mean= np.mean(DistanceROI)
    DistanceROI_std= np.std(DistanceROI)
    psnr_mean=np.mean(psnrlist)
    psnr_std=np.std(psnrlist)
    ssim_mean=np.mean(ssimlist)
    ssim_std=np.std(ssimlist)






