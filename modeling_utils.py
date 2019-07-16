"""
Classes and functions used to train model
"""
import numpy as np
import keras

class NumpyDataGenerator(keras.utils.Sequence):
    """
    Based on original work done by:
        - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Supports only *.npy files
    """
    def __init__(self,
                    list_ids,
                    labels,
                    id_to_file_mapping,
                    batch_size=32,
                    dim=(128,128),
                    n_classes=10,
                    shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_ids = list_ids
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.id_to_file_mapping = id_to_file_mapping
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            self.indexes = np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, id in enumerate(list_ids_temp):
            # Store sample
            _data = np.load(self.id_to_file_mapping[id])[:self.dim[0],:self.dim[1]]
            #only need to pad over the y axis
            y_pad_size = self.dim[1]-_data.shape[1]
            if y_pad_size > 0:
                _data = np.pad(_data, ((0,0),(0,y_pad_size)), 'constant')
            x[i,] = _data
            # Store class
            y[i] = self.labels[i]

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
