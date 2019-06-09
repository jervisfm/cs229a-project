"""
Implements a simple baseline svm classifier for the task of handwritten letter recognition.
"""

import time
import argparse
import os

from sklearn.preprocessing import StandardScaler

import data_loader
import util

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', default=100, help="Number of iterations to perform training.", type=int)
parser.add_argument('--results_folder', default='results/', help="Where to write any results.")
parser.add_argument('--experiment_name', default=None, help="Name for the experiment. Useful for tagging files.")
parser.add_argument('--kernel', default="rbf", help="SVM Kernel. Options are  ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed")

FLAGS = parser.parse_args()

class_file_name = 'class_names_svm'
confusion_file_name = 'confusion_matrix_svm'


def get_suffix_name():
    return "_" + FLAGS.experiment_name if FLAGS.experiment_name else ""

def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format("svm_results", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_confusion_matrix_name():
    return "{}{}".format(confusion_file_name, get_suffix_name())

def main():

    X_train, Y_train = data_loader.get_train(as_vectors=True)
    X_dev, Y_dev = data_loader.get_dev(as_vectors=True)

    print('Scaling and normalizing training data')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print('Normalizing dev data from training transformation')
    X_dev = scaler.transform(X_dev)
    print('Data normalization complete')

    start_time_secs = time.time()
    print("Starting SVM Model training ...", X_train.shape, Y_train.shape)

    classifier = SVC(gamma='auto', verbose=1, kernel=FLAGS.kernel, max_iter=FLAGS.max_iter).fit(X_train, Y_train)

    print("Training done.")
    end_time_secs = time.time()
    training_duration_secs = end_time_secs - start_time_secs
    Y_dev_prediction = classifier.predict(X_dev)

    accuracy = classifier.score(X_dev, Y_dev)

    experiment_result_string = "-------------------\n"
    experiment_result_string += "\nPrediction: {}".format(Y_dev_prediction)
    experiment_result_string += "\nActual Label: {}".format(Y_dev)
    experiment_result_string += "\nAcurracy: {}".format(accuracy)
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format(
        1.0 * training_duration_secs / FLAGS.max_iter)

    class_names = util.get_label(Y_dev)
    classification_report_string = classification_report(Y_dev, Y_dev_prediction, target_names=class_names)
    experiment_result_string += "\nClassification report: {}".format(classification_report_string)

    print(experiment_result_string)

    # Save report to file
    util.write_contents_to_file(get_experiment_report_filename(), experiment_result_string)

    # Generate confusion matrix.
    confusion = confusion_matrix(Y_dev, Y_dev_prediction)

    util.create_confusion_matrices(class_names, confusion, get_confusion_matrix_name())



if __name__ == '__main__':
    main()