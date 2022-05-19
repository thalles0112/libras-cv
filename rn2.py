from hlmsInterface import *
from keras import layers, Sequential
from keras.models import load_model, save_model
from keras import losses
import numpy as np
from hlmsGetter import get_training_data
import os
DIRsmall = '/home/thalles/Documents/small' 
DIR = '/home/thalles/Documents/DATA' 
labels = os.listdir(DIR)

#getandsavedata(DIR=DIR, labels=labels, size=(500,500))

array, labels = readdata()


array = np.array(array[0])
labels = np.array(labels)


print(labels.shape)
#print(labels.shape)

model = Sequential([
    layers.Flatten(),
    layers.Dense(21, 'relu', input_shape=(None,3)),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

model.fit(array, labels, 3, verbose=1, epochs=100)

model.save('modellines_test')