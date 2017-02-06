from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

import helper

def get_model():
    model = Sequential()

    model.add(BatchNormalization(input_shape=(66, 200, 3)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('elu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('elu'))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('elu'))

    model.add(Dense(100))
    model.add(Activation('elu'))

    model.add(Dense(50))
    model.add(Activation('elu'))

    model.add(Dense(10))
    model.add(Activation('elu'))

    model.add(Dense(1))

    return model


def train_model(model, X_train, y_train):
    # Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 128
    model.compile('adam', 'mse')
    history = model.fit(X_train, y_train, nb_epoch=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.2, shuffle=True)
    helper.save_model(model)


if __name__ == '__main__':
    epochs = 5
    batch_size = 128

    model = get_model()
    model.compile('adam', 'mse')

    driving_data = helper.get_data_from_log()
    train, val = helper.train_validation_split(driving_data)

    train_gen = helper.generate_batch(train, batch_size)
    val_gen = helper.generate_batch(val, batch_size)

    samples_per_epoch = len(train) + (batch_size - len(train) % batch_size)
    nb_val_samples = len(val) + (batch_size - len(val) % batch_size)

    history = model.fit_generator(train_gen,
                                  samples_per_epoch=samples_per_epoch,
                                  nb_epoch=epochs,
                                  validation_data=val_gen,
                                  nb_val_samples=nb_val_samples)
    helper.save_model(model)
