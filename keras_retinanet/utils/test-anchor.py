# [anchor_parameters]
# # Sizes should correlate to how the network processes an image, it is not advised to change these!
# sizes   = 32 64 128 256 512
# # Strides should correlate to how the network strides over an image, it is not advised to change these!
# strides = 8 16 32 64 128
# # The different ratios to use per anchor location.
# ratios  = 0.5 1 2 3
# # The different scaling factors to use per anchor location.
# scales  = 1 1.2 1.6

import numpy as np
import keras

base_size = 16

sizes   = [32, 64, 128, 256, 512]
strides = [8, 16, 32, 64, 128]
ratios  = np.array([0.5, 1, 2], keras.backend.floatx())
scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx())
print(ratios)
print(scales)

num_anchors = len(ratios) * len(scales)
print(num_anchors)

# initialize output anchors
anchors = np.zeros((num_anchors, 4))

print(anchors)

# scale base_size
anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
print(anchors)



# compute areas of anchors
areas = anchors[:, 2] * anchors[:, 3]
print(areas)

# correct for ratios
anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
print(anchors)

anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
print(anchors)

# transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
print(anchors)

anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
print(anchors)
