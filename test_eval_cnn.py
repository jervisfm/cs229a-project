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


import data_loader
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
model_filename =  base_model_name + '_model'
model_results_filename =  base_model_name + '_results'

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
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolution layer
    classifier.add(Conv2D(14, (3, 3), activation='relu',  kernel_regularizer=regularizers.l2(l2_reg)))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu',  kernel_regularizer=regularizers.l2(l2_reg)))
    classifier.add(Dense(get_num_classes(), activation='softmax',  kernel_regularizer=regularizers.l2(l2_reg)))
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier



def main():
    print("Evaluating FINAL trained model")
    t0 = time.time()
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # load dataset

    X_train, Y_train = data_loader.get_train(as_vectors=True)
    X_dev, Y_dev = data_loader.get_dev(as_vectors=True)
    X_test, Y_test = data_loader.get_test(as_vectors=True)



    class_names = util.get_label(Y_dev)

    # Reshape to CNN format (M, 28, 28, 1) from (M, 784)
    print("X shape before:", X_train.shape)

    X_train = X_train.reshape((-1, 28, 28, 1))
    print("X shape after: ", X_train.shape)

    print("X dev shape before:", X_dev.shape)
    X_dev = X_dev.reshape((-1, 28, 28, 1))
    print("X dev shape after: ", X_dev.shape)

    print("X test shape before:", X_test.shape)
    X_test = X_test.reshape((-1, 28, 28, 1))
    print("X test shape after: ", X_test.shape)

    print("Done load dataset")


    # encode class values
    encoder = LabelEncoder()
    dummy_y_train = encode_values(encoder, Y_train)
    dummy_y_dev = encode_values(encoder, Y_dev)
    dummy_y_test = encode_values(encoder, Y_test)

    dummy_y_dev_confusion_matrix = encode_values(encoder, Y_dev, forConfusionMatrix=True)
    dummy_y_test_confusion_matrix = encode_values(encoder, Y_test, forConfusionMatrix=True)

    print('Dummy_y (should be one vector of class labels):', dummy_y_train[0])

    print("Done preprocessing dataset")

    # build the model from disk


    model_json_content = open(TEST_EVAL_MODEL, 'r').read()

    loaded_model = model_from_json(model_json_content)
    # load weights into new model
    loaded_model.load_weights(TEST_EVAL_MODEL_WEIGHTS)
    print("Loaded model from disk")

    
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    nn_model = loaded_model

    experiment_result_string = "-------------------\n"



    dummy_y_pred_dev = nn_model.predict(X_dev)
    dummy_y_pred_test = nn_model.predict(X_test)

    # Need to take argmax to find most likely class.
    dummy_y_pred_dev_class = dummy_y_pred_dev.argmax(axis=-1)
    dummy_y_pred_test_class = dummy_y_pred_test.argmax(axis=-1)

    # evaluate the model
    dev_scores = nn_model.evaluate(X_dev, dummy_y_dev, verbose=0)
    test_scores = nn_model.evaluate(X_test, dummy_y_test, verbose=0)

    experiment_result_string += "\nCNN model DEV %s: %.2f%%" % (nn_model.metrics_names[1], dev_scores[1] * 100)
    experiment_result_string += "\nCNN model TEST %s: %.2f%%" % (nn_model.metrics_names[1], test_scores[1] * 100)

    dev_classification_report_string = classification_report(dummy_y_dev_confusion_matrix, dummy_y_pred_dev_class, target_names=class_names)
    experiment_result_string += "\nDEV Classification report: {}".format(dev_classification_report_string)

    test_classification_report_string = classification_report(dummy_y_test_confusion_matrix, dummy_y_pred_test_class,
                                                             target_names=class_names)
    experiment_result_string += "\nTEST Classification report: {}".format(test_classification_report_string)

    print(experiment_result_string)

    util.write_contents_to_file(get_experiment_report_filename(), experiment_result_string)


    conf_matrix = confusion_matrix(dummy_y_test_confusion_matrix, dummy_y_pred_test_class)
    print("Test Confusion matrix", conf_matrix)
    util.create_confusion_matrices(class_names, conf_matrix, get_confusion_matrix_name())
    print ('EVAL DONE')



if __name__ == '__main__':
    main()
