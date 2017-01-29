import csv
import json

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


DRIVING_LOG_FILE = 'driving_log.csv'
JSON_MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'


def load_train_data():
    images = []
    steering_angles = []

    with open(DRIVING_LOG_FILE, 'rt') as f:
        log_reader = csv.reader(f, delimiter=',')
        for row in log_reader:
            images.append(cv2.imread(row[0]))
            steering_angles.append(row[3])

    images = np.array(images)
    steering_angles = np.array(steering_angles)

    return images, steering_angles


def get_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(1))

    return model


def train_model(model, X_train, y_train):
    # Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 128
    model.compile('adam', 'mse', ['accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=5, batch_size=128,
                        validation_split=0.2, shuffle=True)

    # Store model in file
    model_json = model.to_json()
    with open(JSON_MODEL_FILE, 'w') as f:
        json.dump(model_json, f)

    model.save_weights(WEIGHTS_FILE)


if __name__ == '__main__':
    X_train, y_train = load_train_data()
    model = get_model()
    train_model(model, X_train, y_train)
