# Import packages
import pdb
import numpy as np


class NearestNeighbor(object):
    """
    Apply the Nearest Neighbor technique
    """
    def __init__(self):
        pass


    def train(self, data_train, labels_train):
        """
        Parameters
        ----------
        data_train (ndarray): 
            Image data of dimension = num_images x (im_width x im_height x 3)
        labels_train (ndarray):
            Labels of dimension  = 1 x size num_images

        Returns
        -------
        N/A

        """
        # The nearest neighbor classifier
        self.data_train = data_train
        self.labels_train = labels_train


    def predict(self, data_test):
        """
        Parameters
        ----------
        data_test (ndarray): 
            Image data of dimension = num_images x (im_width x im_height x 3)

        Returns
        -------
        labels_test (ndarray):
            Labels of dimension  = 1 x size num_images

        """
        num_test = data_test.shape[0]
        # Make sure that the output type matches the input type
        labels_test = np.zeros(num_test, dtype = self.labels_train.dtype)

        # Loop over all test rows
        for i in range(num_test):
            # Find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.data_train - data_test[i,:]), axis = 1)
            # Get the index with smallest distance
            min_index = np.argmin(distances)
            # Predict the label of the nearest example
            labels_test[i] = self.labels_train[min_index]

        return labels_test
