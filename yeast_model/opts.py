"""
Opts file for loading images - specifies the image sizes and paths to load data from and save weights to.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'

batch_size = 64  # Batch size to use during training - if you run into memory constraints, reduce this
im_h = 64  # Height of input images
im_w = 64  # Width of input images

learning_rate = 1e-4
epochs = 30

# Path to save the weights in after training
checkpoint_path = '/home/nilsahin/PycharmProjects/paired_cell_inpainting/yeast_model/pretrained_weights/'
# Path to get image data
data_path = '/home/nilsahin/crops/Screens_Rad52'

if checkpoint_path != '' and not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
