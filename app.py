from flask import Flask
from flask import render_template
from flask import request
import base64


import numpy as np

from PIL import Image

app = Flask(__name__)

import numpy as np
import pandas
import time
import random
import os
import keras
import datetime

import pickle
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
from keras import regularizers
from training_plot import TrainingPlot

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from keras.models import model_from_json

import util
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_iter', 100, 'Number of steps/epochs to run training.')
flags.DEFINE_integer('batch_size', 500, 'Number of examples to use in a batch for stochastic gradient descent.')
flags.DEFINE_float('l2_reg', 0, 'Amount of L2 regularization to apply to model. Defaults to 0')
flags.DEFINE_string('results_folder', 'results/', 'Folder to store result outputs from run.')
flags.DEFINE_string('experiment_name', 'TEST_FINAL_EVAL', 'Name for the experiment. Useful to tagging files')
flags.DEFINE_integer('model_version', 1,
                     'The version of the model that we want to run. Useful to run different models to compare them')

base_model_name = 'cnn'
model_filename = base_model_name + '_model'
model_results_filename = base_model_name + '_results'

model_weights_filename = base_model_name + '_model_weights'
class_file_name = 'class_names_' + base_model_name
confusion_file_name = 'confusion_matrix_' + base_model_name

TEST_EVAL_MODEL_WEIGHTS = 'results/cnn_model_weights_iter=100_l2reg=0.01'
TEST_EVAL_MODEL = 'results/cnn_model_iter=100_l2reg=0.01'


def get_num_classes():
    # 26 classes for the letters
    return 26


def get_suffix_name():
    return "_" + FLAGS.experiment_name if FLAGS.experiment_name else ""


def get_class_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(class_file_name, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def get_confusion_matrix_name():
    return "{}{}".format(confusion_file_name, get_suffix_name())


def get_confusion_matrix_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(confusion_file_name, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def get_model_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def get_tensorboard_directory():
    tensorboard_root_folder = os.path.join(FLAGS.results_folder, "tensorboard/")
    if not os.path.exists(tensorboard_root_folder):
        os.mkdir(tensorboard_root_folder)

    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    dirpath = os.path.join(tensorboard_root_folder, filename) + "/"

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    now = datetime.datetime.now()
    now_timestring = now.strftime("%Y%m%d_%H%M%S")

    timestamped_dirpath = os.path.join(dirpath, now_timestring)
    if not os.path.exists(timestamped_dirpath):
        os.mkdir(timestamped_dirpath)

    return timestamped_dirpath


def get_training_plots_directory():
    root_folder = os.path.join(FLAGS.results_folder, "training_plots/")
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)

    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    dirpath = os.path.join(root_folder, filename) + "/"

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return dirpath


def get_model_name_only():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    return filename


def get_training_plot_filename():
    directory = get_training_plots_directory()
    return os.path.join(directory, "{}_training_plot".format(get_model_name_only()))


def get_tensorboard_callback(frequency=20):
    logdir = "./" + get_tensorboard_directory()
    print("Tensorboard logdir: ", logdir)
    return keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=frequency,
                                       write_graph=True, write_images=True)


def get_model_weights_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_weights_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_results_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def encode_values(encoder, Y, forConfusionMatrix=False):
    # Use input y to encode
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to dummy variables (i.e. one hot encoded)
    if not forConfusionMatrix:
        # Want 1-hot for training
        one_hot_y = np_utils.to_categorical(encoded_Y)
        return one_hot_y
    else:
        # Want the class labels (numbers) for confusion.
        return encoded_Y


def decode_values(encoder, one_hot_y):
    return encoder.inverse_transform(one_hot_y)


def model(l2_reg=0):
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(
        Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolution layer
    classifier.add(Conv2D(14, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    classifier.add(Dense(get_num_classes(), activation='softmax', kernel_regularizer=regularizers.l2(l2_reg)))
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier


NN_MODEL = None
def get_test_model():
    global NN_MODEL
    if NN_MODEL:
        return NN_MODEL
    print("Evaluating FINAL trained model")
    t0 = time.time()
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # load dataset



    # build the model from disk

    model_json_content = open(TEST_EVAL_MODEL, 'r').read()

    loaded_model = model_from_json(model_json_content)
    # load weights into new model
    loaded_model.load_weights(TEST_EVAL_MODEL_WEIGHTS)
    print("Loaded model from disk")

    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    nn_model = loaded_model

    experiment_result_string = "-------------------\n"

    NN_MODEL = nn_model
    return nn_model

def predict_letter(input_image_data_array):
    X_test = input_image_data_array

    nn_model = get_test_model()

    dummy_y_pred_test = nn_model.predict(X_test)

    # Need to take argmax to find most likely class.
    dummy_y_pred_test_class = dummy_y_pred_test.argmax(axis=-1)
    labels = util.get_label(None)

    print(">>>> FINAL PRED Index: ", dummy_y_pred_test_class)

    pred_label = labels[dummy_y_pred_test_class[0]]
    print(">>>> FINAL PRED Letter: ", pred_label)
    # evaluate the model
    return pred_label



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST', 'GET'])
def classify():
    b64data = request.form['save_remote_data']

    b64data = b64data[22:]
    print(b64data[0:5])
    data = base64.b64decode(b64data)

    with open('test.png', 'wb') as f:
        print('save data')
        f.write(data)

    img = Image.open('test.png').convert('LA')
    img.load()
    img = img.resize((28, 28), Image.ANTIALIAS)
    data_np_arrray = np.asarray(img, dtype="int32")


    np_image_array =  np.zeros((28,28))
    x,y,z = data_np_arrray.shape
    for i in range(x):
        for j in range(y):
            np_image_array[i,j] = data_np_arrray[i,j,1]

    vector = np_image_array.reshape((-1, 1))
    for elemen in vector:
        if elemen > 0:
            print( '>>>greater than 0', elemen)

    #print(data_np_arrray)


    np_image_array = np_image_array.reshape((1, 28, 28, 1))

    pred_letter = predict_letter(np_image_array)


    return render_template('classify.html', letter=pred_letter)
