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


def read_image_file(filename):
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


if __name__ == '__main__':
    DATASET_DIR = '../../datasets/MNIST'

    img_data = read_image_file('%s/training.images' % DATASET_DIR)
    print(img_data['data'])
    print(np.max(img_data['data']))
    print(img_data['num_images'])
    print(img_data['num_rows'])
    print(img_data['num_columns'])

    import matplotlib.pyplot as plt
    for i in range(img_data['data'].shape[0]):
        plt.imshow(img_data['data'][i], cmap='Greys')
        plt.show()