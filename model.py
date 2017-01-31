import csv
import json

import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization


DRIVING_LOG_FILE = 'driving_log.csv'
JSON_MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'


def load_train_data():
    images = []
    steering_angles = []

    with open(DRIVING_LOG_FILE, 'rt') as f:
        log_reader = csv.reader(f, skipinitialspace=True)
        for row in log_reader:
            images.append(cv2.imread(row[0]))
            steering_angles.append(float(row[3]))

    images = np.array(images)
    steering_angles = np.array(steering_angles)

    return images, steering_angles


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

    # Store model in file
    model_json = model.to_json()
    with open(JSON_MODEL_FILE, 'w') as f:
        json.dump(model_json, f)

    model.save_weights(WEIGHTS_FILE)


def preprocess(images):
    images = np.array([cv2.resize(img, (200, 66)) for img in images])
    return images

if __name__ == '__main__':
    X_train, y_train = load_train_data()
    X_train = preprocess(X_train)
    model = get_model()
    model.summary()
    train_model(model, X_train, y_train)
