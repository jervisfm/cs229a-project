

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

from scipy import io as sio



print("Loading matlab  data...")
data = sio.loadmat('data/matlab/emnist-letters.mat')['dataset']
print("Matlab data loaded ")

X_train = data['train'][0,0]['images'][0,0]

y_train = data['train'][0,0]['labels'][0,0]

X_test = data['test'][0,0]['images'][0,0]
y_test = data['test'][0,0]['labels'][0,0]


val_start = X_train.shape[0] - X_test.shape[0]
X_val = X_train[val_start:X_train.shape[0],:]
y_val = y_train[val_start:X_train.shape[0]]
X_train = X_train[0:val_start,:]
y_train = y_train[0:val_start]


X_train = X_train.reshape( (X_train.shape[0], 28, 28), order='F')
X_val = X_val.reshape( (X_val.shape[0], 28, 28), order='F')
X_test = X_test.reshape( (X_test.shape[0], 28, 28), order='F')




def get_train():
    return X_train, y_train

def get_dev():
    return X_val, y_val

def get_test():
    return X_test, y_test



if __name__ == '__main__':
    print('Able to load data ok')
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', y_train.shape)
    print('X_dev shape: ', X_val.shape)
    print('Y_dev shape:', y_val.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', y_test.shape)