import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from sklearn.preprocessing import LabelEncoder
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.layers import Dropout

#from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt

#read training data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.fashion_mnist.load_data()

#Fashion images are the inputs and target variable are 10 classes of different clothing items including accessories

target_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

print("training data dimension :",train_data.shape)
print("testing data dimension: ",eval_data.shape)

# normalise the data (255 is the maximum value in the image matrix)

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)
eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)

# reshape the training data

train_data = train_data.reshape(train_data.shape[0], 28, 28,1)
eval_data = eval_data.reshape(eval_data.shape[0], 28, 28,1)
train_labels = train_labels.reshape(train_data.shape[0],1)
eval_labels = eval_labels.reshape(eval_data.shape[0],1)

print ("number of training examples = " + str(train_data.shape[0]))
print ("number of test examples = " + str(eval_data.shape[0]))
print ("X_train shape: " + str(train_data.shape))
print ("Y_train shape: " + str(train_labels.shape))
print ("X_test shape: " + str(eval_data.shape))
print ("Y_test shape: " + str(eval_labels.shape))

# model architect

model = Sequential()
model.add(Conv2D(40, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
model.add(Conv2D(200, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Conv2D(512, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'  ))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.utils import np_utils

train_labels = np_utils.to_categorical(train_labels).astype('int32')
eval_labels = np_utils.to_categorical(eval_labels)

print ("Y_train shape: " + str(train_labels.shape))
print ("Y_eval shape: " + str(eval_labels.shape))

print("Starting model fitting:")

history = model.fit(train_data, train_labels,
                              validation_data = (eval_data,eval_labels),
                              batch_size=3, epochs = 5,
                              verbose = 1)
