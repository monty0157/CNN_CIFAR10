import pickle

#unpickle CIFAR100 data
#Each data_dict is a dict with a data (b'data') array containing images
# and a label array containing labels (b'lables')
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding="bytes")
    return data_dict
