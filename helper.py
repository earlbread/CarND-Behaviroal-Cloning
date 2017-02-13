import json

import cv2
import numpy as np
import pandas as pd
from scipy.misc import imresize
from sklearn.utils import shuffle


DRIVING_LOG_FILE = 'driving_log.csv'
JSON_MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'


def get_data_from_log():
    log = pd.read_csv(DRIVING_LOG_FILE, skipinitialspace=True)

    return log


def train_validation_split(data, validation_split=0.2):
    def reindex(data):
        return data.reset_index().drop('index', 1)

    mask = np.random.rand(len(data)) < validation_split

    validation = reindex(data[mask])
    train = reindex(data[~mask])

    return train, validation


def get_shuffled_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)


def get_new_image(data, index):
    choice = np.random.randint(0, 3)
    correction = 0.25

    if choice == 0:
        image = cv2.imread(data['center'][index])
        steering = data['steering'][index]
    elif choice == 1:
        image = cv2.imread(data['left'][index])
        steering = data['steering'][index] + correction
    else:
        image = cv2.imread(data['right'][index])
        steering = data['steering'][index] - correction

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image, steering = process_image(image, steering)

    return image, steering


def generate_batch(samples, batch_size=128):
    num_samples = len(samples)

    while True:
        new_samples = get_shuffled_dataframe(samples)
        for offset in range(0, num_samples, batch_size):
            images = []
            steerings = []
            start = offset
            end = offset + batch_size
            end = num_samples if end > num_samples else end

            for i in range(start, end):
                image, steering = get_new_image(new_samples, i)

                images.append(image)
                steerings.append(steering)

            batch_x = np.array(images)
            batch_y = np.array(steerings)

            yield shuffle(batch_x, batch_y)


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
