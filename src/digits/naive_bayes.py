"""Module to test the Naive Bayes implementation and compare with scikit-learn
code.

OBS: Naives Bayes fails when attributes are correlated. It naively assumes the
attributes are independent.
"""


import sys, os
sys.path.append(os.path.abspath('..'))


import datasets


if __name__ == '__main__':
    import sys

    DATASET_DIR = '../../datasets/MNIST'

    args = sys.argv[1 : ]
    training_fname = args[0]
    test_fname = args[1]

    training_img_info = datasets.read_images_file('%s.images' % training_fname)
    training_lb_info = datasets.read_labels_file('%s.labels' % training_fname)

    test_img_info = datasets.read_images_file('%s.images' % test_fname)
    test_lb_info = datasets.read_labels_file('%s.labels' % test_fname)
