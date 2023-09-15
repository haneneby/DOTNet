
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
import shutil
from shutil import rmtree,copyfile,copy2
import zipfile
import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
lgr = logging.getLogger('global')
lgr.setLevel(logging.INFO)
from sklearn.preprocessing import label_binarize


def load_data_t(direc):

     # direc= '/local-scratch/Hanene/Data/multi-freq/Data/'
     print (direc)
     #TESTNSET
     train_dirc1='new_NG_2_2_manysize'
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

     X_test_750,y_test =(testmeasure_750,immatrix_test) 

     return preprocess_t(X_test_750,y_test)



def preprocess_t(X_test_750,y_test):
 
     # y_train= immatrix
     x_test_1= X_test_750
     y_testima= y_test

     # normalize data

     x_test_2= normalize_data(X_test_750) 
     y_testima=y_testima/.25

     y_test = np.reshape(y_testima, (len(y_testima), 128, 128,1))  #


     return  x_test_2, y_test
