import pickle
import os
import numpy as np

#unpickle CIFAR100 data
#Each data_dict is a dict with a data (b'data') array containing images
# and a label array containing labels (b'lables')
def unpickle(directory):
    files = os.listdir(directory)
    dataset = []
    labels = []
    for file in files:
        with open(directory + file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding="bytes")
            dataset.append(data_dict[b'data'])
            labels.append(data_dict[b'labels'])
    return np.asarray(dataset), np.asarray(labels)
