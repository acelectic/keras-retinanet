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

layers = ["P3", "P4", "P5", "P6", "P7"]

layer_outputs = [model.get_layer(layer_name).output for layer_name in layers] 


activation_model = mss.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(image) 

first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 2], cmap='gray')

first_layer_activation = activations[1]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 2], cmap='gray')
plt.show()

# from keras.utils import plot_model
# plot_model(model, to_file='model.png')


# from IPython.display import SVG
# from keras.utils import model_to_dot

# SVG(model_to_dot(model).create(prog='dot', format='svg'))