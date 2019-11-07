#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')



# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import losses

from keras import models as mss
import pprint

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

import keras.backend as K

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)


image = read_image_bgr('examples/pigeon-12_000180.png')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

image = np.expand_dims(image, axis=0)

# ## Load RetinaNet model

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = 'snapshots/model-infer-merge-resnet50-canchor-ep20-loss-0.1881.h5'

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

layer_names = []
# layer_names += ["P3", "P4", "P5", "P6", "P7"]
layer_names += ["clipped_boxes"]

layer_outputs = [model.get_layer(layer_name).output for layer_name in layer_names] 


activation_model = mss.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(image) 


for i, layer in enumerate(activations):
    print(layer_names[i], layer.shape)


first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 2], cmap='gray')
plt.show()
# first_layer_activation = activations[1]
# print(first_layer_activation.shape)
# plt.matshow(first_layer_activation[0, :, :, 2], cmap='gray')


images_per_row = 2

    
# for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
#     # n_features = layer_activation.shape[-1] # Number of features in the feature map
#     n_features = 4 # Number of features in the feature map    
#     h = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
#     w = layer_activation.shape[2] #The feature map has shape (1, size, size, n_features).
#     n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
#     display_grid = np.zeros((h * n_cols, images_per_row * w))
#     for col in range(n_cols): # Tiles each filter into a big horizontal grid
#         for row in range(images_per_row):
#             channel_image = layer_activation[0,
#                                              :, :,
#                                              col * images_per_row + row]
#             print(channel_image.shape)
#             # channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
#             # channel_image /= channel_image.std()
#             # channel_image *= 64
#             # channel_image += 128
#             # channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * h : (col + 1) * h, # Displays the grid
#                          row * w : (row + 1) * w] = channel_image
#     plt.matshow(display_grid, cmap='gray')
#     plt.show()
    # print(display_grid.shape)
    # scale = 1. 
    # plt.figure(figsize=(display_grid.shape[0],
    #                     display_grid.shape[1]))
    # plt.title(layer_name)
    # plt.grid(False)
    # plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    
    
#     plt.matshow(layer[0, :, :, 2], cmap='gray')

# plt.show()

# from keras.utils import plot_model
# plot_model(model, to_file='model.png')


# from IPython.display import SVG
# from keras.utils import model_to_dot

# SVG(model_to_dot(model).create(prog='dot', format='svg'))