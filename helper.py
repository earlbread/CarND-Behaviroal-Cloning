import json

import cv2
import numpy as np
import pandas as pd

DRIVING_LOG_FILE = 'driving_log.csv'
JSON_MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'


def load_train_data():
    images = []
    steering_angles = []

    log = pd.read_csv(DRIVING_LOG_FILE, skipinitialspace=True)

    for i, steering in enumerate(log['steering']):
        images.append(cv2.imread(log['center'][i]))
        images.append(cv2.imread(log['left'][i]))
        images.append(cv2.imread(log['right'][i]))
        steering_angles.append(steering)
        steering_angles.append(steering + 0.25)
        steering_angles.append(steering - 0.25)

    images = np.array(images)
    steering_angles = np.array(steering_angles)

    return images, steering_angles


def crop(image):
    h = 66
    w = 200

    x = int((image.shape[1] - w) / 2)
    y = int((image.shape[0] - h) / 2)

    return image[y:y+h, x:x+w]


def preprocess(images):
    images = np.array([crop(img) for img in images])
    return images


def save_model(model):
    # Store model in file
    model_json = model.to_json()
    with open(JSON_MODEL_FILE, 'w') as f:
        json.dump(model_json, f)

    model.save_weights(WEIGHTS_FILE)
