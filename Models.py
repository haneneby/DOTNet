#Multi-Frequency reconstruction models
from Block_models import *
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

class Models(object):

    def __init__(self, input_shape):
        
        self.input_shape = input_shape

    ##  Multi-Freq_Joint recons and Diag model
    def Recons_model(self):
	    target_shape = [128, 128,1]
	    normal=keras.initializers.he_normal(seed=None)

        ### Add noise 
	    input0 = keras.Input(shape = self.input_shape)
	    input0_n = GaussianNoise(0.1)(input0)

	    model1 = reconst_block(input0_n , 32, initializers=normal, shape= target_shape)
	    out_r = Conv2D(filters = 1, kernel_size = 7, strides = 1,kernel_initializer='glorot_normal', padding = "same", name="reconstruction_output1")(model1)

	    generator_model = Model(inputs = input0,  outputs = out_r)
	    generator_model.summary([])
	    return generator_model

    
