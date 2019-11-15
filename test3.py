import keras
import numpy as np
input1 = keras.layers.Input(shape=(16,),name='In1')
x1 = keras.layers.Dense(8, activation='relu',name='x1')(input1)
input2 = keras.layers.Input(shape=(16,), name='In2')
x2 = keras.layers.Dense(8, activation='relu', name='x2')(input2)
# equivalent to `added = keras.layers.add([x1, x2])`
added = keras.layers.Add(name='add')([input1, input2])
out = keras.layers.Dense(8, name='out')(added)
model = keras.models.Model(inputs=[input1, input2], outputs=added)

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# layer_outputs = [model.get_layer(layer_name).output for layer_name in layer_names] 

# activation_model = mss.Model(inputs=model.input, outputs=layer_outputs)
n1 = np.zeros((16,))
n2 = np.zeros((16,))
n1[2] = 1.
n2[2] = 1.
print(n1,n2)
activations = model.predict([[n1], [n2]]) 


for i, layer in enumerate(activations):
    print(i, layer.shape, layer)