"""Module to test the Naive Bayes implementation and compare with scikit-learn
code.

OBS: Naives Bayes fails when attributes are correlated. It naively assumes the
attributes are independent.
"""


import numpy as np

import sys, os
sys.path.append(os.path.abspath('..'))

import datasets


def calculate_prior_probs(lb_info):
    num_labels = lb_info['num_labels']
    num_label_names = lb_info['num_label_names']
    label_names = lb_info['label_names']
    labels = lb_info['labels']

    prior_probs = np.zeros(lb_info['num_label_names'])

    for i in range(num_label_names):
        prior_probs[i] = len(labels[labels == label_names[i]]) / num_labels

    return prior_probs


class GaussianNB(object):

    def __init__(self, num_classes, prior_probs, feature_dim):
        self.num_classes = num_classes
        self.prior_probs = prior_probs

        self.means = np.zeros((self.num_classes, feature_dim))
        self.variances = np.zeros((self.num_classes, feature_dim))

    def fit(self, features, labels, label_names):
        for i in range(self.num_classes):
            label_name = label_names[i]
            indices = np.where((labels - label_name) == 0)[0]
            flt_features = features[indices]

            self.means[i] = np.mean(flt_features, axis=0)
            self.variances[i] = np.var(flt_features, axis=0)


if __name__ == '__main__':
    DATASET_DIR = '../../datasets/MNIST'

    # training data
    training_img_info = datasets.read_images_file('%s/training.images' % DATASET_DIR)
    training_lb_info = datasets.read_labels_file('%s/training.labels' % DATASET_DIR)

    # test data
    test_img_info = datasets.read_images_file('%s/test.images' % DATASET_DIR)
    test_lb_info = datasets.read_labels_file('%s/test.labels' % DATASET_DIR)

    num_classes = training_lb_info['num_label_names']
    prior_probs = calculate_prior_probs(training_lb_info)
    features = datasets.extract_features(training_img_info)

    gnb = GaussianNB(num_classes, prior_probs, features.shape[1])
    gnb.fit(features, training_lb_info['labels'], training_lb_info['label_names'])