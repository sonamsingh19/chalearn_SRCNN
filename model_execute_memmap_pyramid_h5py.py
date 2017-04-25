#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:20:37 2017

@author: ssingh
"""
import numpy as np
from keras_model_pyramid import get_vesselnet
import sys
import os 
from generate_patches_parallel_pyramid_h5py import patch_size
import h5py
from keras.preprocessing.image import ImageDataGenerator
#num_patches =np.load('tr_patches.npy').shape[0]
#num_patches = 210040
chunk_size =50*100


assert len(sys.argv) ==4 , 'python program unknown/bicubic 2/3/4 tr/valid'
mode = sys.argv[1]
scale= sys.argv[2]

split =sys.argv[3] #train or valid
assert split in ['train','valid']
assert scale in ['2','3','4']
assert mode in ['unknown','bicubic']

print('mode: ' +mode +' scale is:'+scale )
f1=h5py.File('tr_patches.h5','r')
num_patches =f1['data'].shape[0]

tr = f1['data']

#scale='2'
#print('normalized')
f2 = h5py.File(split + '_patches_'+mode+ '_X' + '2'+'.h5','r')#,shape=(num_patches,patch_size, patch_size,3))
tr_x2=f2['data']


#f3=h5py.File(split + '_patches_'+mode+ '_X' + '3'+'.h5','r')#h5py.File('tr_patches.h5','r')
##num_patches =f_1['data'].shape[0]
#
#tr_x3 = f3['data']


f4=h5py.File(split + '_patches_'+mode+ '_X' + '4'+'.h5','r')#h5py.File('tr_patches.h5','r')
#num_patches =f_1['data'].shape[0]

tr_x4 = f4['data']

model =get_vesselnet(3, patch_size/4, patch_size/4)

save_direc = 'models_cnn_'+ mode+'_'+'pyramid_'+'_'+split

if not os.path.exists(save_direc):
      os.mkdir(save_direc)

import keras
#callbac = keras.callbacks.ModelCheckpoint('checkpoint', monitor='train_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


for epochs in range(60):
#  print('total epochs',)    
  for i in range(0,int(num_patches), chunk_size):
        print ( 'epoch:' + str(epochs)+' chunk:'+str(i*1.0/chunk_size)+'/'+str(num_patches/( chunk_size)))
        
        model.fit((tr_x4[i:i+ chunk_size].transpose((0,3,1,2))/255.0),\
                 \
                  (tr[i:i+ chunk_size].transpose((0,3,1,2))/255.0),\
               
                  nb_epoch=1,  \
           batch_size =50, verbose=1)
       
       
  model.save_weights(save_direc+ os.path.sep + str(epochs))

f1.close()
f2.close()



#f3.close()
f4.close()
