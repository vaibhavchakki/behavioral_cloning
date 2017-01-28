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
from PIL import ImageEnhance


new_image_shape = (80, 80)
STEERING_OFFSET = 0.20


def brighten_image(image):
    factor = np.random.uniform(0.75, 1.25)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def process_image_dir():
    paths  = []
    angles = []
    csv = pd.read_csv(os.path.join(os.getcwd(), 'data', 'driving_log.csv'))

    for i, row in csv.iterrows():
        angle = row['steering']
        speed = row['speed']

        left_path = os.path.join(os.getcwd(), 'data', row['left'].strip())
        paths.append(left_path)
        angles.append(angle + STEERING_OFFSET)

        right_path = os.path.join(os.getcwd(), 'data', row['right'].strip())
        paths.append(right_path)
        angles.append(angle - STEERING_OFFSET)

        if (speed < 20.) and angle == 0.:
            continue

        if abs(angle) > 0.3:
            center_path = os.path.join(os.getcwd(), 'data', row['center'].strip())
            paths.append(center_path)
            angles.append(angle)

    return paths, angles


def load_and_process_image(path, angle):
    image = load_img(path, target_size=new_image_shape)
    steering_angle = angle

    if np.random.uniform() < 0.5:
        image = brighten_image(image)

    image = img_to_array(image)

    if np.random.uniform() < 0.5:
        image = flip_axis(image, 1)
        steering_angle = -steering_angle

    #if np.random.uniform() < 0.5:
    #    image = random_shift(image, 0, 0.2, 0, 1, 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = (image/255. - 0.5).astype('float32')

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
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, activation='linear'))

    return model


def training_generator(batch_size, features, labels):

    while True:
        x = []
        y = []

        for i in range(batch_size):
            index = np.random.randint(len(features) - 1)

            image, angle = load_and_process_image(features[index], labels[index])
            x.append(image)
            y.append(angle)

        yield np.array(x), np.array(y)

def train_model():
    #
    # Generate training data
    features, labels = process_image_dir()
    print("Data generate done: %d" % len(features))


    #
    # Get model & compile
    model = get_model(new_image_shape)
    adam = Adam(lr=0.001)
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])

    print("Model compiled")

    model.fit_generator(training_generator(256, features, labels),
                        samples_per_epoch=20480,
                        nb_epoch=4,
                        verbose=1,
                        callbacks=[],
                        validation_data=None)

    with open("model.json", "w") as fp:
        json.dump(model.to_json(), fp)

    model.save_weights("model.h5", overwrite=True)



if __name__ == '__main__':
    train_model()