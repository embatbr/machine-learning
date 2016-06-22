"""Module to test the Naive Bayes implementation and compare with scikit-learn
code.

OBS: Naives Bayes fails when attributes are correlated. It naively assumes the
attributes are independent.
"""


import numpy as np

import sys, os
sys.path.append(os.path.abspath('..'))

import bases


def calculate_prior_probs(lb_info):
    num_labels = lb_info['num_labels']
    num_label_names = lb_info['num_label_names']
    label_names = lb_info['label_names']
    labels = lb_info['labels']

    prior_probs = np.zeros(lb_info['num_label_names'])

    for i in range(num_label_names):
        prior_probs[i] = len(labels[labels == label_names[i]]) / num_labels

    return prior_probs


def accuracy(prediction, labels):
    equals = prediction == labels
    return len(equals[equals]) / len(labels)


class NB(object):

    def __init__(self, num_classes, prior_probs, feature_dim):
        self.num_classes = num_classes
        self.prior_probs = prior_probs

    def fit(self, features, labels, label_names):
        pass

    def predict(self, features_row, label_names):
        pass

    def predict_many(self, features, label_names):
        pass


class MultinomialNB(NB):

    def __init__(self, num_classes, prior_probs, feature_dim):
        super().__init__(num_classes, prior_probs, feature_dim)

        self.likelihoods = np.zeros((self.num_classes, feature_dim))


    def fit(self, features, labels, label_names, alpha=1):
        num_label_names = len(label_names)
        alpha_denom = alpha * features.shape[1]

        for i in range(num_label_names):
            label_name = label_names[i]
            indices = np.where(labels == label_name)[0]
            flt_features = features[indices]

            self.likelihoods[i] = np.sum(flt_features, axis=0) + alpha
            self.likelihoods[i] = self.likelihoods[i] / (np.sum(flt_features) + alpha_denom)


    def predict(self, features_row, label_names):
        indices = np.where(features_row == 1)[0]

        max_index = -1
        max_post_prob = -sys.maxsize

        for i in range(self.num_classes):
            flt_likelihoods = self.likelihoods[i][indices]
            post_prob = np.sum(np.log(flt_likelihoods)) + np.log(self.prior_probs[i])

            if post_prob > max_post_prob:
                max_post_prob = post_prob
                max_index = i

        return label_names[max_index]


    def predict_many(self, features, label_names):
        num_features = len(features)
        prediction = np.empty(num_features)

        for n in range(num_features):
            features_row = features[n]
            prediction[n] = self.predict(features_row, label_names)

        return prediction


class GaussianNB(NB):

    def __init__(self, num_classes, prior_probs, feature_dim):
        self.num_classes = num_classes
        self.prior_probs = prior_probs

        self.means = np.zeros((self.num_classes, feature_dim))
        self.variances = np.zeros((self.num_classes, feature_dim))

    def fit(self, features, labels, label_names):
        for i in range(self.num_classes):
            label_name = label_names[i]
            indices = np.where(labels == label_name)[0]
            flt_features = features[indices]

            self.means[i] = np.mean(flt_features, axis=0)
            self.variances[i] = np.var(flt_features, axis=0)


if __name__ == '__main__':
    DATASET_DIR = '../../datasets/MNIST'

    # training data
    training_img_info = bases.read_MNIST_images('%s/training.images' % DATASET_DIR)
    training_lb_info = bases.read_MNIST_labels('%s/training.labels' % DATASET_DIR)

    # test data
    test_img_info = bases.read_MNIST_images('%s/test.images' % DATASET_DIR)
    test_lb_info = bases.read_MNIST_labels('%s/test.labels' % DATASET_DIR)

    num_classes = training_lb_info['num_label_names']
    prior_probs = calculate_prior_probs(training_lb_info)
    features_training = bases.extract_MNIST_features(training_img_info)
    features_test = bases.extract_MNIST_features(test_img_info)

    labels_training = training_lb_info['labels']
    label_names_training = training_lb_info['label_names']
    labels_test = test_lb_info['labels']
    label_names_test = test_lb_info['label_names']

    mnb = MultinomialNB(num_classes, prior_probs, features_training.shape[1])
    mnb.fit(features_training, labels_training, label_names_training)

    prediction = mnb.predict_many(features_training, label_names_training)
    acc = accuracy(prediction, labels_training)
    print('accuracy (training): %f' % acc)

    prediction = mnb.predict_many(features_test, label_names_test)
    acc = accuracy(prediction, labels_test)
    print('accuracy (test): %f' % acc)
