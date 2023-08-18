
"""
@author: Haneneby
"""
from numpy import genfromtxt
import numpy as np
from Utils.Data_utils import *
import os
import glob
import csv
import pandas as pd
from numpy import *
from Utils.Utils_models import normalize_data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from pathlib import Path
import shutil
from shutil import rmtree,copyfile,copy2
import zipfile
import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
lgr = logging.getLogger('global')
lgr.setLevel(logging.INFO)
from sklearn.preprocessing import label_binarize


def load_data(direc):

     # direc= '/local-scratch/Hanene/Data/multi-freq/Data/'
     print (direc)
    #TRAINSET

     train_dirc1='new_NG_2_2_manysize'
     train_dirc2='new_NG_2_2_manysize_128LED_Many_lesion'
     train_dirc3='new_smallblobs_only_many'
     train_dirc4='new_one_blobs'

     path1 = direc+train_dirc1+'/'+'750/absmat' 
     immatrix1= loadimage(path1)

     path2 =   direc+train_dirc2+'/'+'750/absmat'  
     immatrix2= loadimage(path2)

     path3 =   direc+train_dirc3+'/'+'750/absmat'  
     immatrix3= loadimage(path3)

     path4 =   direc+train_dirc4+'/'+'750/absmat'  
     immatrix4= loadimage(path4)
     
     immatrix= np.concatenate((immatrix1,immatrix2), axis=0)
     immatrix= np.concatenate((immatrix,immatrix3), axis=0)
     immatrix= 100*np.concatenate((immatrix,immatrix4), axis=0)




     path1 =  direc+train_dirc1+'/'+'750/csv'  
     measure1=loadmeasure(path1)

     path2 =  direc+train_dirc2+'/'+'750/csv'  
     measure2=loadmeasure(path2)

     path3 =  direc+train_dirc3+'/'+'750/csv'  
     measure3=loadmeasure(path3)
     
     path4 =  direc+train_dirc4+'/'+'750/csv'  
     measure4=loadmeasure(path4)

     measure_750= np.concatenate((measure1,measure2), axis=0)
     measure_750= np.concatenate((measure_750,measure3), axis=0)
     measure_750= np.concatenate((measure_750,measure4), axis=0)


     #TESTNSET
     #path load GT image
     path1 = direc+train_dirc1+'/'+'testData/absmat' 
     immatrix1= loadimage(path1)

     # path2 =   direc+train_dirc2+'/'+'testData/absmat' 
     # immatrix2= loadimage(path2)
     immatrix_test=  100*immatrix1
     # immatrix_test= 100*np.concatenate((immatrix1,immatrix2), axis=0)
   
     #750 measure
     path1 = direc+train_dirc1+'/'+'testData/csv'
     measure1=loadmeasure(path1)
     # path1 = direc+train_dirc+'/'+'testData/csv'
     # measure2=loadmeasure(path1)
     # testmeasure_750= np.concatenate((measure1,measure2), axis=0)   
     testmeasure_750=measure1
     

     measure_750, immatrix= augmentdata(measure_750,immatrix)

     X_train_750, y_train = shuffle(measure_750, immatrix, random_state=2) 

     X_test_750,y_test =(testmeasure_750,immatrix_test) 

     return preprocess(X_train_750,y_train,X_test_750,y_test)


def preprocess(X_train_750,y_train,X_test_750,y_test):
     

     y_trainima= y_train
     y_testima= y_test

     # normalize data
     x_train_2= normalize_data (X_train_750) 
     x_test_2= normalize_data(X_test_750)
     y_trainima= y_trainima/.25
     y_testima=y_testima/.25
     y_train = np.reshape(y_trainima, (len(y_trainima), 128, 128,1))  
     y_test = np.reshape(y_testima, (len(y_testima), 128, 128,1))  




     return  x_train_2, y_train, x_test_2,  y_test
def augmentdata(measure_750,immatrix):

     reverse_immatrix=immatrix[:,:,::-1] #np.fliplr(immatrix)
     immatrix= np.concatenate((immatrix,reverse_immatrix), axis=0)
     reverse_measure_750=measure_750[...,::-1] 
     measure_750= np.concatenate((measure_750,reverse_measure_750), axis=0)


     return measure_750,immatrix
