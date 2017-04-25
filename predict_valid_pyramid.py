import numpy as np
from keras_model_pyramid import get_vesselnet
from skimage import io
import os
from skimage import measure
import sys

from generate_patches_parallel_pyramid_h5py import patch_size
assert len(sys.argv) ==4 , 'python program unknown/bicubic 2/3/4 epoch'
mode = sys.argv[1]
scale= sys.argv[2]
epoch= sys.argv[3]



assert scale in ['2','3','4']
assert mode in ['unknown','bicubic']



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
            patch_imgs.append(make_patch( img,scale))
      return patch_imgs


def merge_patch(im_height, im_width, patches,scale=1.0):
        
      pred_img = np.zeros((im_height,im_width,3),dtype=np.uint8)
      patch_size_lr= int(patch_size*scale)

      count=0
      for i   in range(0, im_height, patch_size_lr):
           for j in range(0, im_width,patch_size_lr):
                
                   if (i+patch_size_lr) > (im_height):
#                               print('i',i)
                               i =im_height- patch_size_lr
#                               print('shifted',i,j)
                               pred_img[i:i+patch_size_lr,j: j+patch_size_lr]=patches[count]
                               count +=1

                   else:
                         if (j+patch_size_lr) > (im_width):
#                               print('j',j)
                               j= im_width - patch_size_lr
                               pred_img[i:i+patch_size,j: j+patch_size_lr]=patches[count]
                               count +=1

                         else:
                               pred_img[i:i+patch_size_lr,j: j+patch_size_lr]=patches[count]
                
                               count +=1

                
               
            
      return pred_img


                
               
            

model =get_vesselnet(3, patch_size/int(scale), patch_size/int(scale))
#
##save_direc = 'models_cnn_'+ mode+'_'+scale+'_'+split
save_direc = 'models_cnn_'+ mode+'_'+'pyramid'+'__'+'train'
model.load_weights(save_direc+'/'+ epoch)
#
## 'unknown/DIV2K_valid_LR_unknown/X4_interop'
#
out_valid_x2_fol = mode +'/'+'DIV2K_valid_LR_'+mode+'/'+'X'+scale
#
val_x2_bi_files = sorted(os.listdir(out_valid_x2_fol))
#
for i in range(len(val_x2_bi_files[3:4])):
#
      img_x2 = io.imread(out_valid_x2_fol + os.path.sep + val_x2_bi_files[5])
      print(i)
#
from skimage import transform
g= make_patch((img_x2/255.0),1/4.0)

#g=(2*(g/255.0))-1
k=model.predict(g.transpose((0,3,1,2)))
#
#
k=k.transpose((0,2,3,1))
#
#
l = k*255.0
io.imsave('test.png',merge_patch(img_x2.shape[0]*4,img_x2.shape[1]*4,l))
im=io.imread('test.png')
io.imshow(im)
img_true= io.imread('DIV2K_valid_HR/'+val_x2_bi_files[5].split('.')[0][:-2]+'.png')

#print(measure.compare_psnr(img_true,img_x2))

from  skimage import restoration

#import cv2


bi=transform.rescale(img_x2,4.0)
io.imsave('bi.png',bi)
bi =restoration.denoise_bilateral(bi)
bi=io.imread('bi.png')

#io.imshow(new_im)

#print(measure.compare_psnr(img_true,bi))

print(measure.compare_psnr(img_true,im))
print(measure.compare_psnr(img_true,bi))

def predict_valid_x2():
      
      total_time =[]
      out_valid_x2_fol = mode +'/'+'DIV2K_valid_LR_'+mode+'/'+'X'+scale
      pred_valid_x2_fol=mode +'_X'+scale+ '_interop'
      
      if not os.path.exists(pred_valid_x2_fol):
            os.mkdir(pred_valid_x2_fol)
      val_x2_bi_files = sorted(os.listdir(out_valid_x2_fol))

      for i in range(len(val_x2_bi_files[:])):
    
            img_x2 = io.imread(out_valid_x2_fol + os.path.sep + val_x2_bi_files[i])
            print(i)
           
            start=datetime.datetime.now()

            g= make_patch((img_x2/255.0),1/4.0)
            #
            #g=(2*(g/255.0))-1
            k=model.predict(g.transpose((0,3,1,2)))
            #
            #
            k=k.transpose((0,2,3,1))
            #
            #
            l = k*255
            img_x2_pred = merge_patch(img_x2.shape[0]*4,img_x2.shape[1]*4, l)
#            im=io.imread('test.png')
            
            end=datetime.datetime.now()
            print('time per image :x2',(end-start).total_seconds())
            
            total_time.append((end-start).total_seconds())
            print(i,pred_valid_x2_fol +os.path.sep+ val_x2_bi_files[i])
            io.imsave( pred_valid_x2_fol +os.path.sep+ val_x2_bi_files[i],img_x2_pred)
#           
#      print(np.array(psnr).mean())
#runtime per image [s] : 2.1
#
#CPU[1] / GPU[0] : 1
#
#Extra Data [1] / No Extra Data [0] : 0
      with open(pred_valid_x2_fol +os.path.sep+'readme.txt',mode='w')  as f:
            f.writelines(['runtime per image [s] : '+str((sum(total_time)/100.0)+0.5)])
            f.writelines([os.linesep])
            f.writelines([os.linesep])

            f.writelines('CPU[1] / GPU[0] : 0')
            f.writelines(os.linesep)
            f.writelines([os.linesep])

            f.writelines(['Extra Data [1] / No Extra Data [0] :  0'])
#Extra Data [1] / No Extra Data [0] : 0
            
import datetime

#start=datetime.datetime.now()


predict_valid_x2()
#end=datetime.datetime.now()
#print('time per image :x2',(end-start).total_seconds()/(1.0*(100)))
