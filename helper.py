import csv
import json

import cv2
import numpy as np

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


def preprocess(images):
    images = np.array([cv2.resize(img, (200, 66)) for img in images])
    return images


def save_model(model):
    # Store model in file
    model_json = model.to_json()
    with open(JSON_MODEL_FILE, 'w') as f:
        json.dump(model_json, f)

    model.save_weights(WEIGHTS_FILE)
