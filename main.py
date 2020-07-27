# Import outside files
import techniques
# Import packages
import pdb
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np


class ParseData():
    """
    Parse the data
    """
    def __init__(self, data_file, label_file):
        self.data_file = data_file
        self.label_file = label_file
        self.data_dict = {}
        self.label_dict = {}


    def unpickle(self, file_name):
        """
        Parameters: N/A
        Returns: data_dict
        """
        with open(file_name, 'rb') as f:
            return pickle.load(f, encoding='bytes')


    def get_data_values(self):
        """
        Parameters: N/A
        Returns: (batch_label, labels, data, filenames)
        """
        file_data = self.unpickle(self.data_file)
        batch_label = file_data[b'batch_label']
        labels = file_data[b'labels']
        temp_data = file_data[b'data']
        # Format data into a better format (1,3072) -> (32,32,3)
        data = []
        for i in range(temp_data.shape[0]):
            data.append(temp_data[i].reshape((3,-1)).T.reshape((32,32,3)))
        filenames = file_data[b'filenames']
        self.data_dict = {
            'batch_label': batch_label,
            'labels': labels,
            'data': data,
            'filenames': filenames
         }


    def get_label_values(self):
        temp_dict = self.unpickle(self.label_file)
        label_dict = {}
        for i in range(len(temp_dict[b'label_names'])):
            label_dict[i] = temp_dict[b'label_names'][i].decode()
        self.label_dict = label_dict


    def preprocess_data(self):
        """
        """
        self.get_data_values()
        self.get_label_values()


    def plot_image(self, image_num):
        """
        Parameters: image_num - the index of the image to plot
        Returns: N/A
        """
        if ((image_num < 0) or (image_num >= len(self.data_dict['data']))):
            raise Exception('Image number not valid, must be: ({},{})'.format(0,len(self.data_dict['data'])-1))
        plt.imshow(self.data_dict['data'][image_num])
        plt.show()


    def test_nearest_neighbor(self, num_train, num_test):
        """
        Test the Nearest Neighbor technique

        Parameters
        ----------
        num_train (int):
            Number of images to be used to train (typically around 80-90%)
        num_test (int):
            Number of images to be used to train (typically around 10-20%)

        Returns
        -------
        N/A

        """
        # Initialize the Nearest Neighbor class
        nearest_neighbor = techniques.NearestNeighbor()
        # Get the data
        temp_data = self.data_dict['data']
        # Make sure the inputs are valid
        if ((num_train > len(temp_data)) or (num_test > len(temp_data)) or
                ((num_train < 1) or (num_test < 1)) or ((num_train + num_test) > len(temp_data))):
            raise Exception('Subset size not valid, must be: ({},{})'.format(1,len(self.data_dict)-1))
        temp_labels = self.data_dict['labels']
        # Get a random sample of images to train
        train_nums = random.sample(range(len(temp_data)), num_train)
        test_nums = np.setdiff1d(range(len(temp_data)), train_nums)
        # Set the arrays
        data_train = np.array([np.array(temp_data[x]).flatten() for x in train_nums])
        labels_train = np.array([np.array(temp_labels[x]).flatten() for x in train_nums])
        data_test = np.array([np.array(temp_data[x]).flatten() for x in test_nums])
        labels_test = np.array([np.array(temp_labels[x]).flatten() for x in test_nums])
        # Train the classifier
        nearest_neighbor.train(data_train, labels_train)
        # Test the classifier
        labels_pred = nearest_neighbor.predict(data_test)
        # Determine the accuracy
        print('Accuracy: {}%'.format(np.mean(labels_pred == labels_test)))
