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
flags.DEFINE_string('experiment_name', None, 'Name for the experiment. Useful to tagging files')
flags.DEFINE_integer('model_version', 1,
                     'The version of the model that we want to run. Useful to run different models to compare them')

base_model_name = 'cnn'
model_filename =  base_model_name + '_model'
model_results_filename =  base_model_name + '_results'

model_weights_filename = base_model_name + '_model_weights'
class_file_name = 'class_names_' + base_model_name
confusion_file_name = 'confusion_matrix_' + base_model_name


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
    t0 = time.time()
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # load dataset

    X_train, Y_train = data_loader.get_train(as_vectors=True)
    X_dev, Y_dev = data_loader.get_dev(as_vectors=True)

    class_names = util.get_label(Y_dev)

    # Reshape to CNN format (M, 28, 28, 1) from (M, 784)
    print("X shape before:", X_train.shape)
    X_train = X_train.reshape((-1, 28, 28, 1))
    print("X shape after: ", X_train.shape)

    print("X dev shape before:", X_dev.shape)
    X_dev = X_dev.reshape((-1, 28, 28, 1))
    print("X dev shape after: ", X_dev.shape)

    print("Done load dataset")


    # encode class values
    encoder = LabelEncoder()
    dummy_y_train = encode_values(encoder, Y_train)
    dummy_y_dev = encode_values(encoder, Y_dev)

    dummy_y_dev_confusion_matrix = encode_values(encoder, Y_dev, forConfusionMatrix=True)


    print('Dummy_y (should be one vector of class labels):', dummy_y_train[0])

    print("Done preprocessing dataset")

    # build the model
    nn_model = model(l2_reg=FLAGS.l2_reg)

    plot_losses = TrainingPlot(get_training_plot_filename())
    nn_model.fit(X_train, dummy_y_train, epochs=FLAGS.max_iter, batch_size=FLAGS.batch_size, verbose=1,
                  callbacks=[get_tensorboard_callback(), plot_losses], validation_data=(X_dev, dummy_y_dev))

    t1 = time.time()
    training_duration_secs = t1 - t0
    experiment_result_string = "-------------------\n"
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format(
        1.0 * training_duration_secs / FLAGS.max_iter)

    dummy_y_pred_dev = nn_model.predict(X_dev)

    # Need to take argmax to find most likely class.
    dummy_y_pred_dev_class = dummy_y_pred_dev.argmax(axis=-1)

    # evaluate the model
    scores = nn_model.evaluate(X_dev, dummy_y_dev, verbose=0)
    experiment_result_string += "CNN model DEV %s: %.2f%%" % (nn_model.metrics_names[1], scores[1] * 100)

    classification_report_string = classification_report(dummy_y_dev_confusion_matrix, dummy_y_pred_dev_class, target_names=class_names)
    experiment_result_string += "\nClassification report: {}".format(classification_report_string)

    print(experiment_result_string)

    util.write_contents_to_file(get_experiment_report_filename(), experiment_result_string)

    # serialize model to JSON
    # TODO: make this configurable.
    model_json = nn_model.to_json()
    with open(get_model_filename(), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    nn_model.save_weights(get_model_weights_filename())
    print("Saved model to disk")

    print("Dummy y pred dev class", dummy_y_pred_dev_class[0])
    conf_matrix = confusion_matrix(dummy_y_dev_confusion_matrix, dummy_y_pred_dev_class)
    print("Confusion matrix", conf_matrix)
    util.create_confusion_matrices(class_names, conf_matrix, get_confusion_matrix_name())



if __name__ == '__main__':
    main()