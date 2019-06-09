import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Dimension for the input feature bitmap vectors.
BITMAP_DIM = 784

label_mapping = {
    1 : 'A',
    2 : 'B',
    3 : 'C',
    4 : 'D',
    5 : 'E',
    6 : 'F',
    7 : 'G',
    8 : 'H',
    9 : 'I',
    10 : 'J',
    11 : 'K',
    12 : 'L',
    13 : 'M',
    14 : 'N',
    15 : 'O',
    16 : 'P',
    17 : 'Q',
    18 : 'R',
    19 : 'S',
    20 : 'T',
    21 : 'U',
    22 : 'V',
    23 : 'W',
    24 : 'X',
    25 : 'Y',
    26: 'Z',
}

def convert_emnist_label_to_letter(numeric_label):
    return label_mapping[numeric_label]

def get_label(Y):

    labels = label_mapping
    labels_indices= []
    for key,value in labels.items():
        labels_indices.append(value)

    print('Labels: ', labels)
    print('Labels Indices (1-based):', labels)

    return labels_indices

def create_confusion_matrices(class_names, confusion, file_name):
    np.set_printoptions(precision=2)

    figsize = (8*2, 6*2)
    # Plot non-normalized confusion matrix
    fig1 = plt.figure(figsize=figsize)
    plot_confusion_matrix(confusion, classes=class_names,
                          title='Confusion matrix, without normalization')
    fig1.savefig('imgs/cm_imgs/' + file_name + '.png')
    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=figsize)
    plot_confusion_matrix(confusion, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    fig2.savefig('imgs/cm_imgs/' + file_name + '_norm.png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          font_size=8):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=font_size)
    plt.yticks(tick_marks, classes, fontsize=font_size)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=font_size,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def write_contents_to_file(output_file, input_string):
    with open(output_file, 'w') as file_handle:
        file_handle.write(input_string)