
This repo contains code for super resolution intended for competition at NTIRE  CVPR 2017 workshop `http://www.vision.ee.ethz.ch/ntire17/ <http://www.vision.ee.ethz.ch/ntire17/>`_ workshop hosted at `codalab link <https://competitions.codalab.org/competitions/16308#results>`_

Summary
------------------

**Two type of models are explored:**

1. Upsample (Upsample in between)

.. image:: https://raw.githubusercontent.com/sonamsingh19/chalearn_SRCNN/master/images/scale4.png

2. SRCNN (first bicubic and then CNN)

.. image:: https://raw.githubusercontent.com/sonamsingh19/chalearn_SRCNN/master/images/scale3.png

**Results**

For Track : unknown (unknown degrading factors):
**X4: 25.33 PSNR** (validation) which puts in Top-10 in the leaderboard with ``model#2``. (patch size=30)
 
*Unfortunately:* For bicubic track, I wasn't able to perform better than bicubic baseline. I guess GAN needs to be explored.


The only modification required to use your own model is to change *keras_model_pyramid.py*.

Execution
------------------
**File Structure**

- download.sh : links for downloading datasets
- generate_patches_parallel_pyramid_h5py.py: generate patches and bicubic interpolations 
- keras_model_pyramid.py : keras model (model#2) for scale X4 (2 upsample)
- model_execute_memmap_pyramid_h5py.py : model to be excuted using h5py for batches over saved patches.
- predict_valid_pyramid.py : predict over validation images
- evaluate_valid.py : To quickly  evaluate for PSNR between images stored in different folder (validation)

**Directory Structure for Execution**

For downloaded directory with name SRCNN: Training data needs to be stored in this strcture:

- SRCNN/model_execute_memmap_pyramid_h5py.py
- SRCNN/DIV2K_train_HR
- SRCNN/unknown/DIV2K_train_LR_unknown/X4
- SRCNN/unknown/DIV2K_train_LR_unknown/X3
- SRCNN/unknown/DIV2K_valid_LR_unknown/X4


**Run**

For Training:
for unknown:
``python model_execute_memmap_pyramid_h5py.py unknown 4 train``

for bicubic:
``python model_execute_memmap_pyramid_h5py.py bicubic 4 train``

For Prediction:
``python predict_valid_pyramid.py bicubic 4``

Some of the boilerplate code (PSNR loss etc.) has been borrowed from SRGAN repo: https://github.com/dribnet/srgan
