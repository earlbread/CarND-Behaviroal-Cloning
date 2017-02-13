import json

import cv2
import numpy as np
import pandas as pd
from scipy.misc import imresize


DRIVING_LOG_FILE = 'driving_log.csv'
JSON_MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'


def get_data_from_log():
    log = pd.read_csv(DRIVING_LOG_FILE, skipinitialspace=True)

    return log


def get_new_image(data):
    index = np.random.randint(len(data))
    choice = np.random.randint(0, 3)

    if choice == 0:
        image = cv2.imread(data['center'][index])
        steering = data['steering'][index]
    elif choice == 1:
        image = cv2.imread(data['left'][index])
        steering = data['steering'][index] + 0.25
    else:
        image = cv2.imread(data['right'][index])
        steering = data['steering'][index] - 0.25

    image, steering = process_image(image, steering)

    return image, steering


def train_validation_split(data, validation_split=0.2):
    def reindex(data):
        return data.reset_index().drop('index', 1)

    mask = np.random.rand(len(data)) < validation_split

    validation = reindex(data[mask])
    train = reindex(data[~mask])

    return train, validation


def generate_batch(data, batch_size=128):
    while True:
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            image, steering = get_new_image(data)

            batch_x.append(image)
            batch_y.append(steering)
        yield np.array(batch_x), np.array(batch_y)


def crop_and_resize(image, top=60, bottom=25, size=(64, 64)):
    """
    After crop top and bottom, resize image.
    """
    row = image.shape[0]
    cropped = image[top:row-bottom, :]
    resized = imresize(cropped, size)
    return resized


def random_translate(image):
    px = int(image.shape[1] / 10)
    py = int(image.shape[0] / 10)
    x = np.random.randint(-px, px)
    y = np.random.randint(-py, py)
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image,M, (image.shape[1], image.shape[0]))


def random_flip(image, steering_angle, prob=0.5):
    if np.random.rand() < prob:
        return np.fliplr(image), -steering_angle
    else:
        return image, steering_angle


def process_image(image, steering):
    image = crop_and_resize(image)
    image = random_translate(image)
    image, steering = random_flip(image, steering)

    return image, steering
