"""
Model definition and script
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Reshape, Conv2D, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.optimizers import Adam
from modeling_utils import NumpyDataGenerator
import sys

def train_model(metadata_location,
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
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
    print(model.summary())
    # Train model on dataset
    steps_per_epoch = np.ceil(len(metadata) / batch_size)
    validation_steps = np.ceil(len(list(test_data['classID']))/batch_size)
    # checkpoint
    filepath="./keras_checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6,
                        verbose=1,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        epochs=1,
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
    metadata_location = sys.argv[1]
    file_base_location = sys.argv[2]
    train_model(metadata_location, file_base_location)
