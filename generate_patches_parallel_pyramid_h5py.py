#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:50:31 2017

@author: ssingh
"""

import os
from skimage import measure
from skimage import transform
from skimage import io
import numpy as np
import datetime 
import sys
from joblib import  Parallel,delayed
import h5py
patch_size=120

def make_patch(x,scale):
#      x shape (2040, 1608, 3) varies
      patches=[]
      patch_size_lr= int(patch_size*scale)

      for i in range(0, x.shape[0], patch_size_lr):
             for j in range(0, x.shape[1],patch_size_lr):

                   if i+patch_size_lr > (x.shape[0]):
#                               print('i',i)
                               i =x.shape[0]- patch_size_lr
#                               print('shifted',i,j)
                               patches.append(x[i:i+patch_size_lr, j:j+patch_size_lr])
                         
                   elif j+patch_size_lr > (x.shape[1]):
#                               print('j',j)
                               j=x.shape[1]-patch_size_lr
                               patches.append(x[i:i+patch_size_lr,j: j+patch_size_lr])
                   else:
                              patches.append(x[i:i+patch_size_lr,j: j+patch_size_lr])
                        
                         
                              
      return np.array(patches)


def collect_patches(imgs,scale=1.0):
      patch_imgs =[]
      for img in imgs:
            patch_imgs.append(make_patch(img,scale))
      return patch_imgs

if __name__=='__main__':
              
      tr_fol ='DIV2K_train_HR'
      tr_files = sorted(os.listdir(tr_fol))
      
      unknown ='unknown'
      bicubic = 'bicubic'
      tr_paths ={'unknown':[ 'DIV2K_train_LR_unknown/X2',\
                'DIV2K_train_LR_unknown/X3',\
                'DIV2K_train_LR_unknown/X4'],
                 'bicubic':[ 'DIV2K_train_LR_bicubic/X2',\
                'DIV2K_train_LR_bicubic/X3',\
                'DIV2K_train_LR_bicubic/X4']}
      
      valid_paths={'unknown':[ 'DIV2K_valid_LR_unknown/X2',\
                'DIV2K_valid_LR_unknown/X3',\
                'DIV2K_valid_LR_unknown/X4'],
                 'bicubic':[ 'DIV2K_valid_LR_bicubic/X2',\
                'DIV2K_valid_LR_bicubic/X3',\
                'DIV2K_valid_LR_bicubic/X4']}
      
      
      
      print ('bicubic interpolation for train datasets')
      
      scales={'x2':2,'x3':3, 'x4':4}
      
      
      def bicubic_interop(f,p,im_name,out_path):
               
              img = io.imread(f + os.path.sep+ p+ os.path.sep + im_name)
            
              start=datetime.datetime.now()
            
              transform_scale = scales[im_name.split('.')[0][-2:]]
              new_img =  transform.rescale( img, transform_scale)
              
              
              end=datetime.datetime.now()
              print('time per image for scale'+ str(transform_scale),(end-start).total_seconds())
              io.imsave(out_path +os.path.sep+ im_name, new_img)
              
      
      for f in tr_paths:
          for p in tr_paths[f]:
             out_path= f+os.path.sep+p+'_interop'
             
             if not os.path.exists(out_path):
                   os.mkdir(out_path)
                   
             print ('reading from: '+ f+os.path.sep+p + ' saving to: '+ out_path)
             files = (os.listdir(f+os.path.sep+p))
             
            
             Parallel(n_jobs=15)(delayed(bicubic_interop)(f,p,im_name,out_path) for im_name in files)
                    
      
      print ('bicubic interpolation for validation datasets')
             
      for f in valid_paths:
          for p in valid_paths[f]:
             out_path= f+os.path.sep+p+'_interop'
             
             if not os.path.exists(out_path):
                   os.mkdir(out_path)
                   
             print ('reading from: '+ f+os.path.sep+p + ' saving to: '+ out_path)
             files = (os.listdir(f+os.path.sep+p))
             Parallel(n_jobs=15)(delayed(bicubic_interop)(f,p,im_name,out_path) for im_name in files)
      
      #       for im_name in files:
      #              img = io.imread(f + os.path.sep+ p+ os.path.sep + im_name)
      #              transform_scale = scales[im_name.split('.')[0][-2:]]
      #              new_img =  transform.rescale( img, transform_scale)
      #              
      #              io.imsave(out_path +os.path.sep+ im_name, new_img)
      
      '''
      Generate patches 
      '''
      
      
      print('generating HR training patches')
      tr_imgs ={}
      for i in range(len(tr_files)):
            img = io.imread(tr_fol + os.path.sep + tr_files[i])
            
      #      img_x2 = io.imread(tr_x2_fol_bi + os.path.sep + tr_x2_files[i])
            
            tr_imgs[tr_files[i]]=img
      #      tr_imgs_x2 [tr_x2_files[i]] = img_x2
      
      d =collect_patches([tr_imgs[k] for k in sorted(tr_imgs.keys())])
      tr_im = np.concatenate(d)
      
      import h5py
      f=h5py.File('tr_patches.h5','w')
      f.create_dataset('data',data=tr_im)
      f.close()
      
      np.save('tr_patches',tr_im)
      
      print('train HR patches saved')
      
      
      print('Generating train LR patches')
      
      for f in sorted(tr_paths,reverse=True):
          for p in sorted(tr_paths[f],reverse=True):
             tr_imgs_interop = {}
             out_path= f+os.path.sep+p
             
      
      
                   
             print ('reading from: '+ out_path)
             files = sorted(os.listdir(f+os.path.sep+p))
             
             for im_name in files:
                    
                    
                    im = io.imread(out_path + os.path.sep+ im_name)
                    tr_imgs_interop[im_name] = im
      
             d2 =collect_patches([tr_imgs_interop[k] for k in sorted(files)],scale=1/(float(p[-1])))
             tr_im_patched = np.concatenate(d2)
             
             h5file=h5py.File('train_patches_'+f+'_'+p.split('/')[-1]+'.h5','w')
             
             print(tr_im_patched.shape)
      
             h5file.create_dataset('data',data=tr_im_patched)
             h5file.close()
      
