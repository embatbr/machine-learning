"""This module reads the binary MNIST files and convert to CSV and NumPy array.
"""


import numpy as np
import struct


BYTE_SIZE_IN_BITS = 8
BYTE_UNIT = 1
INTEGER_SIZE_IN_BYTES = 4 * BYTE_UNIT


def bytes_to_int(bstr):
    ret = (bstr[0] << (3*BYTE_SIZE_IN_BITS)) + (bstr[1] << (2*BYTE_SIZE_IN_BITS))
    ret = ret + (bstr[2] << BYTE_SIZE_IN_BITS) + bstr[3]

    return ret


def read_images_file(filename):
    f = open(filename, 'rb')

    f.read(2 * BYTE_UNIT)
    datatype = f.read(BYTE_UNIT)
    if datatype != b'\x08':
        raise Exception('Data type is not unsigned byte.')

    num_dim = ord(f.read(BYTE_UNIT))

    # (num_imgs, num_rows, num_columns)
    shape = [0] * num_dim
    num_pixels = 1
    for i in range(num_dim):
        shape[i] = bytes_to_int(f.read(INTEGER_SIZE_IN_BYTES))
        num_pixels = num_pixels * shape[i]

    # img with values 0 (all white)
    data = np.zeros(num_pixels, dtype=np.uint8)
    for px in range(num_pixels):
        data[px] = ord(f.read(BYTE_UNIT))

    data = np.reshape(data, shape)

    return {
        'num_images' : shape[0],
        'num_rows' : shape[1],
        'num_columns' : shape[2],
        'data' : data
    }


def read_labels_file(filename):
    f = open(filename, 'rb')

    f.read(2 * BYTE_UNIT)
    datatype = f.read(BYTE_UNIT)
    if datatype != b'\x08':
        raise Exception('Data type is not unsigned byte.')

    num_dim = ord(f.read(BYTE_UNIT))

    # (num_labels)
    shape = [0] * num_dim
    num_labels = 1
    for i in range(num_dim):
        shape[i] = bytes_to_int(f.read(INTEGER_SIZE_IN_BYTES))
        num_labels = num_labels * shape[i]

    labels = np.zeros(num_labels, dtype=np.uint8)
    for lb in range(num_labels):
        labels[lb] = ord(f.read(BYTE_UNIT))

    labels = np.reshape(labels, shape)

    return {
        'num_labels' : shape[0],
        'num_label_names' : 10,
        'labels' : labels,
        'label_names' : list(range(0, 10))
    }


def group_images_by_label(img_info, lb_info):
    if img_info['num_images'] != lb_info['num_labels']:
        raise Exception('Number of images and labels must be equal.')

    img_shape = (1, img_info['num_rows'], img_info['num_columns'])
    grouped_imgs = [None] * lb_info['num_label_names']

    for i in range(img_info['num_images']):
        label = lb_info['labels'][i]
        img = img_info['data'][i]

        if grouped_imgs[label] is None:
            grouped_imgs[label] = np.zeros(img_shape)
            grouped_imgs[label][0] = grouped_imgs[label][0] + img
        else:
            grouped_imgs[label] = np.append(grouped_imgs[label], [img], axis=0)

    return grouped_imgs


def extract_features(img_info):
    num_images = img_info['num_images']
    num_rows = img_info['num_rows']
    num_columns = img_info['num_columns']
    data = img_info['data']

    features = np.zeros((num_images, num_rows*num_columns))#, dtype=np.uint8)

    for i in range(num_images):
        flat_data = np.reshape(data[i], num_rows*num_columns)
        features[i] = features[i] + np.where(flat_data == 0, 0, 1)

    return features


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    DATASET_DIR = '../../datasets/MNIST'

    img_info = read_images_file('%s/training.images' % DATASET_DIR)
    lb_info = read_labels_file('%s/training.labels' % DATASET_DIR)
    grouped_imgs = group_images_by_label(img_info, lb_info)
    features = extract_features(img_info)

    num_rows = img_info['num_rows']
    num_columns = img_info['num_columns']

    for i in range(img_info['num_images']):
        print('label: %d' % lb_info['labels'][i], features[i])

        plt.imshow(img_info['data'][i], cmap='Greys', interpolation='nearest')
        features_reshaped = np.reshape(features[i], (num_rows, num_columns))
        plt.figure()
        plt.imshow(features_reshaped, cmap='Greys', interpolation='nearest')

        plt.show()