# Module blocks for building Multi-freq model

from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
import keras  as keras
from keras.layers import *
from keras.layers import GlobalAveragePooling2D, Reshape, multiply, Permute
from keras import backend as K
from keras.utils.vis_utils import plot_model
import tensorflow as tf



def res_block_gen(model, kernal_size, filters, strides):

    rec = model

    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, kernel_initializer='glorot_normal',padding = "same")(model)
    model = Activation('tanh')(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, kernel_initializer='glorot_normal',padding = "same")(model)
    model = Activation('tanh')(model)
    model = add([rec, model])

    return model

def reconst_block(input, filters, initializers, shape):
    ## Create an initial image estimate''' 
    model= Dense( 128*128, activation = 'relu', kernel_initializer=initializers)(input)
    model = keras.layers.Reshape(shape)(model)
    model = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Activation('relu')(model) 
    for index in range(2): 
        model = res_block_gen(model, 3, 32, 1)
    for index in range(2): 
        model = res_block_gen(model, 5, 32, 1)
    model= Conv2D(filters = 32, kernel_size = 7, strides = 1, padding = "same",kernel_regularizer=tf.keras.regularizers.L2(0.001))(model)
    model = Activation('relu')(model)
    return model


