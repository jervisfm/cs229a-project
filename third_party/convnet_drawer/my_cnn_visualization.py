from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from matplotlib_util import save_model_to_file
from keras_util import convert_drawer_model
import config

 # Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a second convolution layer
classifier.add(Conv2D(14, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(26, activation='softmax'))
# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

config.inter_layer_margin = 25
config.text_size=8

model = convert_drawer_model(classifier)
save_model_to_file(model, "cnn.pdf")
