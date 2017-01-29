import pandas as pd
import numpy as np
import os, sys
import cv2
import matplotlib.pyplot as plt
from scipy import misc
import math
import json

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, Convolution2D, MaxPooling2D, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
from keras.optimizers import Adam
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL.ImageEnhance import Brightness, Color, Contrast, Sharpness


new_image_shape = (80, 80)
STEERING_OFFSET = 0.25

def crop_image(image, enable = 1):
    if enable:
        return image.crop((0, 20, image.size[0], image.size[1] - 20))
    else:
        return image


def normalize(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = (image / 255. - 0.5).astype('float32')
    return image


def apply_modify(image, function, low = 0.5, high = 1.5):
    factor = np.random.uniform(low, high)
    enhancer = function(image)
    return enhancer.enhance(factor)


def transform_shift(image, angle, wrg = 0.2, hrg = 0.0):
    h, w = image.size
    tx = np.random.uniform(-wrg, wrg) * w
    ty = np.random.uniform(-hrg, hrg) * h
    angle = angle + (tx / h) * 0.2
    image = np.asarray(image)
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty]])
    image = cv2.warpAffine(image, translation_matrix, (h, w))
    return image, angle


def preprocess_test_image(image):
    image = crop_image(image)
    image = image.resize(new_image_shape)
    image = img_to_array(image)
    image = normalize(image)
    return image


def preprocess_train_image(image):
    image = crop_image(image)
    image = image.resize(new_image_shape)
    image = apply_modify(image, Brightness)
    image = apply_modify(image, Sharpness, 1., 2.)
    return image


def process_image_dir():
    paths  = []
    angles = []
    csv = pd.read_csv(os.path.join(os.getcwd(), 'data', 'driving_log.csv'))

    for i, row in csv.iterrows():
        angle = row['steering']
        speed = row['speed']

        if (speed <= 20.0):
            continue

        left_path = os.path.join(os.getcwd(), 'data', row['left'].strip())
        paths.append(left_path)
        angles.append(angle + STEERING_OFFSET)

        right_path = os.path.join(os.getcwd(), 'data', row['right'].strip())
        paths.append(right_path)
        angles.append(angle - STEERING_OFFSET)

        #if np.random.uniform() < 0.5:
        center_path = os.path.join(os.getcwd(), 'data', row['center'].strip())
        paths.append(center_path)
        angles.append(angle)

    return paths, angles


def load_and_process_image(path, angle):
    image = load_img(path)
    image = preprocess_train_image(image)
    steering_angle = angle

    transform_value = np.random.randint(3)
    if transform_value == 0:
        image = img_to_array(image)
        image = flip_axis(image, 1)
        steering_angle = -steering_angle
    elif transform_value == 1:
        # vertical shift
        image, angle = transform_shift(image, steering_angle, 0.0, 0.2)
        #image = random_shift(image, 0, 0.2, 0, 1, 2)
    else:
        # horizontal shift
        image, steering_angle = transform_shift(image, steering_angle, 0.2, 0)

    image = normalize(image)

    return image, steering_angle


def get_model(shape):

    model = Sequential()
    #model.add(BatchNormalization(mode=2, axis=1, input_shape=shape))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2),
                            input_shape=(shape[0], shape[1], 3)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    return model


def validation_generator(test_data, test_labels):
    while True:
        index = np.random.randint(len(test_data) - 1)
        x = load_img(test_data[index])
        x = preprocess_test_image(x)
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        y = np.array([[test_labels[index]]])

        yield x, y


def training_generator(batch_size, train_data, train_labels):

    while True:
        x = []
        y = []

        for i in range(batch_size):
            index = np.random.randint(len(train_data) - 1)

            image, angle = load_and_process_image(train_data[index], train_labels[index])
            x.append(image)
            y.append(angle)

        yield np.array(x), np.array(y)


def train_model():
    #
    # Generate training data
    features, labels = process_image_dir()
    print("Data generate done: %d" % len(features))

    #
    # Generating training and testing data
    features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.2,
                                                                                random_state=42)
    #
    # Get model & compile
    model = get_model(new_image_shape)
    adam = Adam(lr=0.001)
    model.compile(loss='mse', optimizer='adam')

    print("Model compiled")

    model.fit_generator(training_generator(256, features_train, labels_train),
                        samples_per_epoch=20480,
                        nb_epoch=4,
                        verbose=1,
                        callbacks=[],
                        validation_data=validation_generator(features_test, labels_test),
                        nb_val_samples=len(features_test))

    with open("model.json", "w") as fp:
        json.dump(model.to_json(), fp)

    model.save_weights("model.h5", overwrite=True)


if __name__ == '__main__':
    train_model()