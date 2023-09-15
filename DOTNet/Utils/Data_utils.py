"""
@author: Haneneby
"""
from numpy import genfromtxt
import numpy as np
import os
import glob
import csv
import pandas as pd
from numpy import *
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

def readcomplex(strng):
	strng = strng.replace(" ", '')
	strng = strng.replace("i", 'j')
	c = 0
	try:
		c = complex(strng)
	except ValueError as e:
		print("Exception on input")
		print(strng)
	return c
def norm(image):
    norm_image = cv2.normalize(image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

def sortfiles(listfiles):
    return [ x for (_,x) in sorted( ( (int(stri[stri.find('-')+1:stri.find('.')]),stri) for stri in listfiles )

                                   ) ]
def loadmeasure(path_to_parent):

    measlist = sortfiles(os.listdir(path_to_parent))
    my_data = (np.vectorize(lambda t:readcomplex(t))(genfromtxt(os.path.join(path_to_parent, csv_file), delimiter=',',dtype='str')) for csv_file in measlist)
    measure = np.absolute(list(my_data))
    # print (measure.shape)
    return  measure

def loadimage(path):
    imlist = removedotfile(path)
    immatrix= array([(np.around(genfromtxt(os.path.join(path, im), delimiter=',',dtype='float'),decimals=4))#.flatten()
                   for im in imlist],'f') 
    # print (immatrix.shape)
    return immatrix 

def nean_std_data(x):
    mvec = x.mean(0)
    stdvec = x.std(axis=0) 
    return mvec, stdvec#,mvec,stdvec
def loadrealmeas(path):
     data = pd.read_csv(path, header=None)
     values = data.values[:, :]
     rmeas = np.asmatrix(values, dtype='float64')    
     return rmeas
def removedotfile (path):
    listing = (os.listdir(path))
    i=0
    while((listing[i]=='.') or (listing[i]=='..') or (listing[i]=='.DS_Store' )):
        i=i+1
    imlist = sortfiles(listing[i:])
    return imlist