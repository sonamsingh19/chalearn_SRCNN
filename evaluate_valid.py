import numpy as np
from skimage import io
import os
from skimage import measure
import sys


out_valid_x2_fol_interop ='bicubic_X4_interop'
files = [f for f in os.listdir(out_valid_x2_fol_interop) if f.endswith('png') ]
psnrs=[]
ssim=[]
for f in sorted(files):
      img =io.imread(out_valid_x2_fol_interop+'/'+f)
      img_true =io.imread('DIV2K_valid_HR/'+f.split('.')[0][:-2]+'.png')
      psnrs.append (measure.compare_psnr(img_true,img))
      print(f)
print('Avg. PSNR',np.array(psnrs).mean())

      
   