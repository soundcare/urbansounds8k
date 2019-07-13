"""
INCOMPLETE, NOT READY TO BE RUN
"""
import numpy as np
from keras.models import Sequential
from modeling_utils import DataGenerator


# Parameters

def train_model(metadata_location,
                model_architecture = None,
                data_dim=(128,128),
                batch_size=64,
                n_classes=10,
                n_channels=1,
                shuffle=False):
    """
    """
    params = {'dim': data_dim,
              'batch_size': batch_size,
              'n_classes': n_classes,
              'n_channels': n_channels,
              'shuffle': shuffle}

    # Datasets
    partition = # IDs
    labels = # Labels

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    # Design model
    model = Sequential()
    [...] # Architecture
    model.compile()

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6)
