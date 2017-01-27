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
from keras.optimizers import Adam
import sklearn.metrics as metrics


new_img_width  = 80
new_img_height = 80
STEERING_OFFSET = 0.30

#
# crop the image 20 pixels from top and 25 pixels from bottom
def crop(img):
    img = img[20:135,:,:]
    return img

#
# convert image from RGB to HSV
def rgb2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#
# read image from path
def load_image(path):
    return cv2.imread(path)

#
# flip image
def flip_image(img):
    return cv2.flip(img, 1)

#
# normalize the image data
def normalize(img):
    img = img.astype('float32')
    img = img/255.0 - 0.5
    return img

def resize(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def process_image_dir():
    paths  = []
    angles = []
    csv = pd.read_csv(os.path.join(os.getcwd(), 'data', 'driving_log.csv'))

    for i, row in csv.iterrows():
        angle = row['steering']

        left_path = os.path.join(os.getcwd(), 'data', row['left'].strip())
        paths.append(left_path)
        angles.append(angle - STEERING_OFFSET)

        right_path = os.path.join(os.getcwd(), 'data', row['right'].strip())
        paths.append(right_path)
        angles.append(angle + STEERING_OFFSET)

        if abs(angle) > 0.3:
            center_path = os.path.join(os.getcwd(), 'data', row['center'].strip())
            paths.append(center_path)
            angles.append(angle)

    return paths, angles


def load_and_process_image(path, angle):
    image = load_image(path)
    image = crop(image)
    image = resize(image, new_img_width, new_img_height)
    steering_angle = angle

    if np.random.randint(2) % 2 == 0:
        image = flip_image(image)
        steering_angle = -steering_angle

    image = normalize(image)

    return image, steering_angle


def model(load, shape):

    model = Sequential()
    #model.add(BatchNormalization(mode=2, axis=1, input_shape=shape))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), input_shape=shape))
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

    model.compile(loss='mse', optimizer="adam")
    return model


def get_model():
    model = Sequential()

    #
    # Modeling NVIDIA network here, add Batchnormalization
    model.add(BatchNormalization(mode=2, axis=1, input_shape=(new_img_width, new_img_height, 3)))

    #
    # Convolution layer 1
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    #                        input_shape=(new_img_width, new_img_width, 3)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #
    # Convolution layer 2
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #
    # Convolution layer 3
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #
    # Convolution layer 4
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    #model.add(MaxPooling2D(pool_size=(1, 1)))

    #
    # Convolution layer 5
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    #model.add(MaxPooling2D(pool_size=(1, 1)))

    #
    # Flatten
    model.add(Flatten())

    #
    # Dense layer 1
    model.add(Dense(1164, activation='relu'))

    #
    # add some dropout layer to avoid overfitting
    #model.add(Dropout(0.5))

    #
    # Dense layer 2
    model.add(Dense(100, activation='relu'))

    #
    # add some dropout layer to avoid overfitting
    #model.add(Dropout(0.5))

    #
    # Dense layer 3
    model.add(Dense(50, activation='relu'))

    #
    # add some dropout layer to avoid overfitting
    #model.add(Dropout(0.5))

    #
    # Dense layer 4
    model.add(Dense(10, activation='relu'))

    #
    # add some dropout layer to avoid overfitting
    #model.add(Dropout(0.5))

    #
    # Output layer
    model.add(Dense(1, activation='relu'))

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

def main():
    #
    # Generate training data
    features, labels = process_image_dir()
    print("Data generate done: %d" % len(features))


    #
    # Get model & compile
    model = get_model()
    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])

    print("Model compiled")

    model.fit_generator(training_generator(128, features, labels),
                        samples_per_epoch=8192,
                        nb_epoch=2,
                        verbose=1,
                        callbacks=[],
                        validation_data=None)

    with open("model.json", "w") as fp:
        json.dump(model.to_json(), fp)

    model.save_weights("model.h5", overwrite=True)



if __name__ == '__main__':
    main()


