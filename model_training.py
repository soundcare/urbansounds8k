"""
Model definition and script
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Reshape, Conv2D, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import regularizers, optimizers
from keras.optimizers import Adam
from modeling_utils import NumpyDataGenerator
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_generator_multiple(generator,directories, batch_size, img_height,img_width):
    generators =[]
    for directory in directories:
        gen = generator.flow_from_directory(directory,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=True,
                                          seed=7)

        generators.append(gen)

    for gen in generators:
        for data, labels in gen:
            yield data, labels

def train_model_from_png(metadata,
                        file_base_location,
                        training_folds = [10,2,3,4,5,6,7,8],
                        validation_folds = [9],
                        batch_size = 128,
                        img_height=128,
                        img_width = 128,
                        approx_fold_size = 8000,
                        nclass = 10):
    # Datasets
    data_dim  = (img_height,img_width)
    metadata = pd.read_csv(metadata_location)
    train_data = metadata[metadata['fold'].isin(training_folds)].reset_index()
    test_data = metadata[metadata['fold'].isin(validation_folds)].reset_index()
    print("Training on {}, validation on {}".format(train_data['fold'].unique(),test_data['fold'].unique()))
    # Generators
    datagen = ImageDataGenerator(rescale=1./255)
    #generators:
    training_generator = datagen.flow_from_dataframe( dataframe=train_data,
                                                    directory=file_base_location,
                                                    x_col="location_png",
                                                    y_col="class_name",
                                                    subset="training",
                                                    batch_size=batch_size,
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode="categorical",
                                                    target_size=data_dim)
    validation_generator = datagen.flow_from_dataframe( dataframe=test_data,
                                                    directory=file_base_location,
                                                    x_col="location_png",
                                                    y_col="class_name",
                                                    subset="validation",
                                                    batch_size=batch_size,
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode="categorical",
                                                    target_size=data_dim)
    print("Indices: {}".format(validation_generator.class_indices))
    model = Sequential()
    model.add(Conv2D(24, (5,5),
                        data_format='channels_last',
                        activation='relu',input_shape=(img_height,img_width,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (5,5),activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu',
                   kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(10, activation='softmax',
                   kernel_regularizer=regularizers.l2(0.001)))
    # Compile model
    model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-3),loss="categorical_crossentropy",metrics=["accuracy"])
    print(model.summary())
    filepath="./keras_checkpoints/png-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    steps_per_epoch = len(list(train_data['classID'])) // batch_size
    validation_steps = len(list(test_data['classID'])) // batch_size
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        verbose=1,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps
                        )


def train_model_from_npy(metadata_location,
                file_base_location,
                model_architecture=None,
                data_dim=(128,128),
                batch_size=64,
                n_classes=10,
                training_folds = [10,2,3,4,5,6,7,8],
                validation_folds = [9],
                shuffle=True):
    """
    """
    params = {'dim': data_dim,
              'batch_size': batch_size,
              'n_classes': n_classes,
              'shuffle': shuffle}

    # Datasets
    metadata = pd.read_csv(metadata_location)
    if shuffle:
        metadata = metadata.sample(frac=1).reset_index()
    id_to_file_mapping = dict(zip(metadata['fsID'],metadata['location']))
    labels = dict(zip(metadata['fsID'],metadata['classID']))
    train_data = metadata[metadata['fold'].isin(training_folds)].reset_index()
    test_data = metadata[metadata['fold'].isin(validation_folds)].reset_index()
    print(train_data['fold'].unique(),test_data['fold'].unique())
    # Generators
    training_generator = NumpyDataGenerator(list(train_data['fsID']), labels,id_to_file_mapping,file_base_location, **params)
    validation_generator = NumpyDataGenerator(list(test_data['fsID']),labels, id_to_file_mapping,file_base_location, **params)
    # Design model
    if model_architecture:
        model = model_architecture
    else:
        model = Sequential()
        model.add(Conv2D(24, (5,5),
                            data_format='channels_last',
                            activation='relu',input_shape=(128,128,1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(48, (5,5),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(48, (5,5),activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu',
                       kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(10, activation='softmax',
                       kernel_regularizer=regularizers.l2(0.001)))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.rmsprop(lr=0.0005, decay=1e-3), metrics=['accuracy'])
    print(model.summary())
    # Train model on dataset
    steps_per_epoch = np.ceil(len(metadata) / batch_size)
    validation_steps = np.ceil(len(list(test_data['classID']))/batch_size)
    # checkpoint
    filepath="./keras_checkpoints/npy-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        #use_multiprocessing=True,
                        #workers=6,
                        verbose=1,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        epochs=10,
                        callbacks=callbacks_list,
                        shuffle=True
                       )
    #Validate
    validation_generator_2 = NumpyDataGenerator(list(test_data['fsID']), labels, id_to_file_mapping, **params)

    predictions = model.predict_generator(validation_generator_2,
                                      steps = validation_steps,
                                      verbose=True)
    y_pred = np.argmax(predictions, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(list(test_data['classID']), y_pred[:len(list(test_data['classID']))])
    print(cm)
    print(np.unique(np.array(y_pred), return_counts=True))
    print(np.unique(np.array(test_data['classID']), return_counts=True))


if __name__ == "__main__":
    model_run = sys.argv[1]
    if model_run == "npy":
        metadata_location = sys.argv[2]
        file_base_location = sys.argv[3]
        train_model_from_npy(metadata_location, file_base_location)
    elif model_run == "png":
        metadata_location = sys.argv[2]
        file_base_location = sys.argv[3]
        train_model_from_png(metadata_location, file_base_location)
