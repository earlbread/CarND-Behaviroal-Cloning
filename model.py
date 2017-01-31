from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

import helper

def get_model():
    model = Sequential()

    model.add(BatchNormalization(input_shape=(66, 200, 3)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

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
    X_train, y_train = helper.load_train_data()
    X_train = helper.preprocess(X_train)
    model = get_model()
    model.summary()
    train_model(model, X_train, y_train)
