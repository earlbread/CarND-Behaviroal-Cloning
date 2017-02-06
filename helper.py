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
            image = process_image(image)

            steering = data['steering'][index]

            batch_x.append(image)
            batch_y.append(steering)
        yield np.array(batch_x), np.array(batch_y)


def crop(image):
    h = 66
    w = 200

    x = int((image.shape[1] - w) / 2)
    y = int((image.shape[0] - h) / 2)

    return image[y:y+h, x:x+w]


def equalizeHist(image):
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    return image


def process_image(image):
    image = crop(image)
    image = equalizeHist(image)

    return image


def save_model(model):
    # Store model in file
    model_json = model.to_json()
    with open(JSON_MODEL_FILE, 'w') as f:
        json.dump(model_json, f)

    model.save_weights(WEIGHTS_FILE)
