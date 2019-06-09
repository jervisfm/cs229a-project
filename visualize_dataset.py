import data_loader
import numpy as np
import matplotlib.pyplot as plt
import util
import random

def get_sample_indices(X_train, y_train, test_label, num_samples=10):
    result = []
    for idx, label in enumerate(y_train):
        if label == test_label:
            if len(result) < 10:
                result.append(idx)
            else:
                break
    return result

def main():
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    fig = plt.figure()

    print ('Loading data...')
    X_train, y_train = data_loader.get_train(as_vectors=False)
    print('Load complete')

    print('Selecting sample letter ...')
    get_sample_indices(X_train, y_train, 3)
    print('Done')
    print('Plotting selected image samples... ')
    num_examples = 10
    labels = list(range(1,27))
    for y_label in labels:
        sample_indices = get_sample_indices(X_train, y_train, y_label, num_samples=num_examples)
        for iteration_index,train_example_index in enumerate(sample_indices):
            a = fig.add_subplot(num_examples, len(labels), iteration_index * len(labels) + y_label)
            # gray_r so that we get gray scale image output
            plt.imshow(X_train[train_example_index], cmap='gray_r')
            #plt.imshow(X_train[train_example_index])
            plt.axis('off')
            if iteration_index == 0:
                label_letter = util.convert_emnist_label_to_letter(y_label)
                a.set_title('{}'.format(label_letter))

    plt.show()
    image_path = 'imgs/emnist_train_visualization.png'
    fig.savefig(image_path)
    print('Visualization saved to ', image_path)

if __name__ == '__main__':
    main()