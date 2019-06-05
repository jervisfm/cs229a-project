import numpy as np
import pandas
import time
import random
import os
import keras
import datetime
import scipy

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


from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19

from keras.layers import Dense

from keras.optimizers import Adam
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
flags.DEFINE_string('results_folder', 'results/', 'Folder to store result outputs from run.')
flags.DEFINE_string('experiment_name', None, 'Name for the experiment. Useful to tagging files')
flags.DEFINE_integer('model_version', 1,
                     'The version of the model that we want to run. Useful to run different models to compare them')
flags.DEFINE_string('transfer_model', 'InceptionV3', 'Model use for transfer learning')
flags.DEFINE_boolean('tune_source_model', False, 'If True, tune the weights of the original model as well')


base_model_name = 'transfer_learning'
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


def transfer_learning(X, y, X_dev, y_dev, source_model=InceptionV3(weights='imagenet', include_top=False), tune_source_model=True):
    # Based on code snippets from:https://keras.io/applications/
    print("Building transfer learning model...")
    # create the base pre-trained model
    base_model = source_model

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # Let's reduce to 128 as that's what gave us good perf in the simple model.
    x = Dense(128, activation='relu')(x)
    #x = Dense(1024, activation='relu')(x)
    # and a logistic softmax layer for output
    predictions = Dense(get_num_classes(), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)

    model.compile(optimizer=Adam(clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    print("Tuning our last custom layer...")
    plot_losses = TrainingPlot(get_training_plot_filename() + "_tune_our_last_layer")
    model.fit(X, y, epochs=FLAGS.max_iter, batch_size=FLAGS.batch_size, verbose=1, validation_data=(X_dev, y_dev), callbacks=[plot_losses])

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    if not tune_source_model:
        return model
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print("Tuning the last 2 inceptions layers ...")
    plot_losses = TrainingPlot( get_training_plot_filename() + "_tune_transfer_model_last_layer")
    model.fit(X, y, epochs=FLAGS.max_iter, batch_size=FLAGS.batch_size, verbose=1,  validation_data=(X_dev, y_dev), callbacks=[plot_losses])
    return  model

def convertTo3Channels(x, new_image_size=(299, 299)):
    print('Convert data to shape: {}'.format(new_image_size))
    # x has shape (m. size, size, # channels)
    num_examples, size_x, size_y, _ = x.shape
    result = np.zeros((num_examples, new_image_size[0], new_image_size[1], 3))
    num_channels = 3

    for i in range(num_examples):
        source_image = x[i, :, :, 0]
        scaled_image = scipy.misc.imresize(source_image, new_image_size)
        for j in range(num_channels):
            result[i, :, :, j] = scaled_image
    return result


# for inline mutate x instead of

def assertXIsNotNan(X):
    if np.isnan(X).any():
        raise ValueError("Oh no input is nan!!!")


def get_source_model(source_model_name):
    print('Using base model: {}'.format(source_model_name))

    if source_model_name == 'MobileNet':
        new_image_size = (224, 224)
        source_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    elif source_model_name == 'InceptionV3':
        new_image_size = (299, 299)
        source_model = InceptionV3(weights='imagenet', include_top=False)
    elif source_model_name == 'ResNet50':
        new_image_size = (224, 224)
        source_model = ResNet50(weights='imagenet', include_top=False)
    elif source_model_name == 'VGG19':
        new_image_size = (224, 224)
        source_model = VGG19(weights='imagenet', include_top=False)
    else:
        raise AssertionError(
            "Please use one of the available models only. They are 'MobileNet', 'InceptionV3', 'ResNet50', 'VGG19'")

    return new_image_size, source_model

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



    new_image_size, source_model = get_source_model(FLAGS.transfer_model)

    print("Converting input images for transfer learning...")
    X_dev = convertTo3Channels(X_dev, new_image_size)
    X_train = convertTo3Channels(X_train, new_image_size)
    print ("Done preprocessing dataset")

    nn_model = transfer_learning(X_train, dummy_y_train, X_dev, dummy_y_dev, source_model, FLAGS.tune_source_model)


    plot_losses = TrainingPlot(get_training_plot_filename())
    nn_model.fit(X_train, dummy_y_train, epochs=FLAGS.max_iter, batch_size=FLAGS.batch_size, verbose=1,
                  callbacks=[plot_losses], validation_data=(X_dev, dummy_y_dev))

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
    experiment_result_string += "Transfer learning model DEV %s: %.2f%%" % (nn_model.metrics_names[1], scores[1] * 100)

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