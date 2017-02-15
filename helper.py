import cv2
import numpy as np
import pandas as pd

from scipy.misc import imresize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

DRIVING_LOG_FILE = './data/driving_log.csv'


def get_data_from_log():
    """Reads driving log file as Pandas dataframe.
    """
    log = pd.read_csv(DRIVING_LOG_FILE, skipinitialspace=True)

    return log


def train_validation_split(data, validation_split=0.2):
    """Splits train and validation set and reset index.
    """
    train, validation = train_test_split(data, test_size=validation_split)

    train.reset_index(drop=True, inplace=True)
    validation.reset_index(drop=True, inplace=True)

    return train, validation


def get_shuffled_dataframe(df):
    """Shuffles data and reset index.
    """
    return df.sample(frac=1).reset_index(drop=True)


def get_new_image(data, index):
    """Reads a image from data corresponding to the index.

    Chooses randomly either left, center, or right image and process image.
    """
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
    """Reads images from samples by batch size and returns it.
    """
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


def random_brightness(image, brightness=0):
    """Adjusts randomly brightness of the image.
    """
    if brightness == 0:
        brightness = np.random.uniform(0.15, 2.0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    hsv[:, :, 2] = np.where(v * brightness > 255, 255, v * brightness)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def random_translate(image, steering):
    """Moves the image randomly.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    px = int(image.shape[1] / 10)
    py = int(image.shape[0] / 10)

    x = np.random.uniform(-px, px)
    y = np.random.uniform(-py, py)
    steering = steering + (x / px * 0.4)

    M = np.float32([[1, 0, x], [0, 1, y]])

    return cv2.warpAffine(image, M, (cols, rows)), steering


def random_flip(image, steering_angle, prob=0.5):
    """Flips the image randomly.
    """
    if np.random.rand() < prob:
        return np.fliplr(image), -steering_angle
    else:
        return image, steering_angle


def process_image(image, steering):
    """Preprocesses on the image.
    """
    image = crop_and_resize(image)
    image = random_brightness(image)
    image, steering = random_translate(image, steering)
    image, steering = random_flip(image, steering)

    return image, steering
