import numpy as np
import pickle
import matplotlib.pyplot as plt

class ParseData():
    """
    Parse the data
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_dict = {}
        self.dict_values = {}

    def unpickle(self):
        """
        Parameters: N/A
        Returns: data_dict
        """
        with open(self.file_name, 'rb') as fo:
            self.data_dict = pickle.load(fo, encoding='bytes')
    
    def get_data_from_dict(self):
        """
        Parameters: N/A
        Returns: (batch_label, labels, data, filenames)
        """
        batch_label = self.data_dict[b'batch_label']
        labels = self.data_dict[b'labels']
        temp_data = self.data_dict[b'data']
        # Format data into a better format (1,3072) -> (32,32,3)
        data = {}
        for i in range(temp_data.shape[0]):
            data[str(i)] = temp_data[i].reshape((3,-1)).T.reshape((32,32,3))
        filenames = self.data_dict[b'filenames']
        self.dict_values = {
            'batch_label': batch_label,
            'labels': labels,
            'data': data,
            'filenames': filenames
         }

    def plot_image(self, image_num):
        """
        Parameters: image_num - the index of the image to plot
        Returns: N/A
        """
        if ((image_num < 0) or (image_num >= len(self.dict_values['data']))):
            raise Exception('Image number not valid, must be: ({},{})'.format(0,len(self.dict_values['data'])-1))
        plt.imshow(self.dict_values['data'][str(image_num)])
        plt.show()

    def preprocess_data(self):
        """
        """
        self.unpickle()
        self.get_data_from_dict()
