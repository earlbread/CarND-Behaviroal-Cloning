import json

import cv2
import numpy as np
import pandas as pd


DRIVING_LOG_FILE = 'driving_log.csv'
JSON_MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'


def get_data_from_log():
    log = pd.read_csv(DRIVING_LOG_FILE, skipinitialspace=True)

    frames = [log['center'], log['left'], log['right']]
    images = pd.concat(frames).reset_index()[0]

    frames = [log['steering'], log['steering'] + 0.25, log['steering'] - 0.25]
    steerings = pd.concat(frames).reset_index()['steering']

    df = pd.DataFrame({'image': images, 'steering': steerings})

    return df


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
            index = np.random.randint(len(data))

            image = cv2.imread(data['image'][index])
            steering = data['steering'][index]

            image, steering = process_image(image, steering)

            batch_x.append(image)
            batch_y.append(steering)
        yield np.array(batch_x), np.array(batch_y)


def crop(image):
    h = 66
    w = 200

    x = int((image.shape[1] - w) / 2)
    y = int((image.shape[0] - h) / 2)

    return image[y:y+h, x:x+w]


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
    image = crop(image)
    image = random_translate(image)
    image, steering = random_flip(image, steering)

    return image, steering


def save_model(model):
    # Store model in file
    model_json = model.to_json()
    with open(JSON_MODEL_FILE, 'w') as f:
        json.dump(model_json, f)

    model.save_weights(WEIGHTS_FILE)
