
import math
import numpy as np
import ConfigParser
#from help_functions import *
from keras.models import Sequential
from keras.layers import Input, merge, Convolution2D,Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.optimizers import SGD,rmsprop,Adam
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, AveragePooling2D, Convolution2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Permute,Reshape
from keras.preprocessing.image import ImageDataGenerator
import sys
from keras.layers import merge
from keras.layers.convolutional import ZeroPadding2D,Cropping2D
import matplotlib.pyplot as plt
import keras
from keras import layers  
def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))




import theano.tensor as T
#def char(y_true,y_pred):
#     return T.abs_(y_true-y_pred).mean() #+ 0.1*(y_true**2+0.001**2)**2

def mae (y_true,y_pred) :
     return T.abs_(y_true-y_pred).mean()  

def get_vesselnet(n_ch,patch_height,patch_width):
    border_mode='same'
    inputs = Input((n_ch, patch_height, patch_width))
#    conv1=ZeroPadding2D((10,10))(inputs)
    c1 = Convolution2D(8,( 3, 3), activation='elu',data_format='channels_first',  
  border_mode=border_mode)(inputs)
    
    c1 = Convolution2D(16, (3,3), activation='elu',data_format='channels_first',   border_mode=border_mode)(c1)
    for i in range(1):
              c1 = merge((Convolution2D(16, (5,5), activation='elu',  \
                                        border_mode=border_mode,data_format='channels_first')(c1),c1,inputs),mode='concat',concat_axis=1)
#    out1 = UpSampling2D(size=(2, 2),data_format='channels_first')(c1)
    out1 = Convolution2DTranspose(16,kernel_size=(2,2),strides=(2,2),data_format='channels_first')(c1)
#    out1=BatchNormalization()(out1)

    out1 = Convolution2D(16, (3,3), activation='elu',data_format='channels_first',   border_mode=border_mode)(out1)

#    get_out1= Convolution2D(3, 3, 3, activation='tanh',   border_mode=border_mode)(out1)
#    out1=BatchNormalization()(out1)
#
    c1 = Convolution2D(16, (3,3), activation='elu',data_format='channels_first',   border_mode=border_mode)(out1)
    for i in range(1):
              c1 = merge((Convolution2D(16, (3 ,3), activation='elu',\
                                        border_mode=border_mode,data_format='channels_first')(c1),c1,out1),mode='concat',concat_axis=1)
    out2 = Convolution2DTranspose(16,kernel_size=(2,2),strides=(2,2),data_format='channels_first')(c1)
#    out2=BatchNormalization()(out2)
    out2 = Convolution2D(16, (3,3), activation='elu',data_format='channels_first',   border_mode=border_mode)(out2)
    out2 = Convolution2D(16, (3,3), activation='elu',data_format='channels_first',   border_mode=border_mode)(out2)
    out2 = Convolution2D(16, (3, 3), activation='elu',data_format='channels_first',   border_mode=border_mode)(out2)

 

    get_out2= Convolution2D(3, (3,3), activation='sigmoid',data_format='channels_first',  border_mode=border_mode)(out2)



    model = Model(inputs=inputs, outputs=[get_out2])

    rms=rmsprop(lr=0.001)


    model.compile(optimizer=rms , loss='mae')#,metrics=[char])#,callback=[callbac])



    return model

