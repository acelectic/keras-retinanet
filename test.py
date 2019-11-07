#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras

import sys
sys.path.insert(0, '../')

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


# ## Load RetinaNet model

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
# model_path = 'snapshots/model-infer-merge-resnet50-canchor-ep20-loss-0.1881.h5'
model_path = 'snapshots/resnet50_20_loss-0.1881_val-loss-1.1923_mAP-0.8562.h5'


# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


# ## Run detection on example
# load image
image = read_image_bgr('examples/pigeon-12_000180.png')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()

# model.summary()
inp = model.input                                           # input placeholder
# outputs = [layer.output for layer in model.layers]  
# boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
# image = np.expand_dims(image, axis=0)


outputs = [layer.output for layer in model.layers] 
functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function

# Testing

layer_outs = functor([image, 1.])
print (layer_outs)

# p3 = model.get_layer('P3').output
# p4 = model.get_layer('P4').output
# p5 = model.get_layer('P5').output
# p6 = model.get_layer('P6').output
# p7 = model.get_layer('P7').output
# fc = K.function([inp, K.learning_phase()], [p3, p4, p5, p6, p7])
# p3_out = fc([image, 1.])[0]
# print(p3_out.shape)

# ff = model.layers[5].output
# fc = K.function([inp, K.learning_phase()], [ff])
# f_out = func([image, 1.])[0]
# print(f_out.shape)

# plt.figure(figsize=(15, 15))
# plt.axis('off')

# for i in model.layers[1:2]:
#     try:
#         # print(i, i.output)
#         out = i.output
#         # print(out)
#         func = K.function([inp, K.learning_phase()], [out])
#         f_out = func([image, 1.])[0]
#         print(f_out.shape)
#         # pprint.pprint(f_out[0])
        
#         dist = f_out[0]
#         dist1 = cv2.convertScaleAbs(dist)
#         dist2 = cv2.normalize(dist, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
#         cv2.imshow("dist", dist)
#         cv2.imshow("dist1", dist1)
#         cv2.imshow("dist2", dist2)
#         cv2.waitKey()
#         # plt.imshow(f_out[0])
#         # K.function([inp, K.learning_phase()], [out])
#     except Exception as e:
#         print("{t}{e}{t}".format(e=e, t='\n\n\n'))



# plt.show()

# print("processing time: ", time.time() - start)

# # correct for image scale
# boxes /= scale

# # visualize detections
# for box, score, label in zip(boxes[0], scores[0], labels[0]):
#     # scores are sorted so we can break
#     if score < 0.5:
#         break
        
#     color = label_color(label)
    
#     b = box.astype(int)
#     draw_box(draw, b, color=color)
    
#     caption = "{} {:.3f}".format(labels_to_names[label], score)
#     draw_caption(draw, b, caption)
    
