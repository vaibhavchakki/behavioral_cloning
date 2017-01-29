#
# Import python packages needed for the Behavirol Cloning Project
#
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

#
# Set the size of new reshaped image and the constant steering angle
# added to the images during training
new_image_shape = (80, 80)
STEERING_OFFSET = 0.25

#
# Define a method to crop an image. This crops the top and bottom row
# 20 pixels
def crop_image(image, enable = 1):
    if enable:
        return image.crop((0, 20, image.size[0], image.size[1] - 20))
    else:
        return image

#
# Method to normalize the data by converting to YUV format and then
# divide by 255, and subtract 0.5
def normalize(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = (image / 255. - 0.5).astype('float32')
    return image

#
# Generic method to apply some image transformation like
# Brightness, Color, Contrast, Sharpness
def apply_modify(image, function, low = 0.5, high = 1.5):
    factor = np.random.uniform(low, high)
    enhancer = function(image)
    return enhancer.enhance(factor)

#
# Method defined to shift image either horizontally or vertically
def random_shift_mod(image, angle, wrg = 0.0, hrg = 0.0):
    h, w = image.size
    tx = np.random.uniform(-wrg, wrg) * w
    ty = np.random.uniform(-hrg, hrg) * h
    angle = angle + (tx / h) * 0.4
    image = np.asarray(image)
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty]])
    image = cv2.warpAffine(image, translation_matrix, (h, w))
    return image, angle

#
# Preproces image passed to the validation Generator
# 1. Crop the image
# 2. Resize
# 3. Normalize
def preprocess_test_image(image):
    image = crop_image(image)
    image = image.resize(new_image_shape)
    image = img_to_array(image)
    image = normalize(image)
    return image

#
# Preproces image passed to the testing Generator
# 1. Crop the image
# 2. Resize
# 3. Brighten, Sharpen
# 4. More processing done in load_and_process_image
def preprocess_train_image(image):
    image = crop_image(image)
    image = image.resize(new_image_shape)
    image = apply_modify(image, Brightness)
    image = apply_modify(image, Sharpness, 1., 2.)
    return image

#
# Read the driving_log.csv and load the data for further processing
# as part of the testing/validation generator
# Each element has the data in below format:
#   image: [left_img, center_img, right_img]
#   angle: [angle + offset, angle, angle - offset]
def process_image_dir():
    paths  = []
    angles = []
    csv = pd.read_csv(os.path.join(os.getcwd(), 'data', 'driving_log.csv'))

    for i, row in csv.iterrows():
        angle = row['steering']
        speed = row['speed']
        path = os.path.join(os.getcwd(), 'data')

        # Ignore data below speed of 20.
        if (speed <= 20.0):
            continue

        images = [path + '/' + row['left'].strip(),
                  path + '/' + row['center'].strip(),
                  path + '/' + row['right'].strip()]

        angle = [angle + STEERING_OFFSET,
                 angle,
                 angle - STEERING_OFFSET]

        paths.append(images)
        angles.append(angle)

        #select_img = np.random.randint(2)

        #if (select_img == 0):
        #    left_path = os.path.join(os.getcwd(), 'data', row['left'].strip())
        #    paths.append(left_path)
        #    angles.append(angle + STEERING_OFFSET)
        #elif select_img == 1:
        #    right_path = os.path.join(os.getcwd(), 'data', row['right'].strip())
        #    paths.append(right_path)
        #    angles.append(angle - STEERING_OFFSET)
        #else:
            #if np.random.uniform() < 0.5:
        #    center_path = os.path.join(os.getcwd(), 'data', row['center'].strip())
        #   paths.append(center_path)
        #    angles.append(angle)

    return paths, angles


#
# Top level method for loading the image and preprocessing
# steps
def load_and_process_image(path, angle):
    image = load_img(path)
    image = preprocess_train_image(image)
    steering_angle = angle

    transform_value = np.random.randint(4)
    if transform_value == 0:
        image = img_to_array(image)
        image = flip_axis(image, 1)
        steering_angle = -steering_angle
    elif transform_value == 1:
        # vertical shift
        image, angle = random_shift_mod(image, steering_angle, 0.0, 0.2)
        #image = random_shift(image, 0, 0.2, 0, 1, 2)
    elif transform_value == 2:
        # horizontal shift
        image, steering_angle = random_shift_mod(image, steering_angle, 0.2, 0)
    else:
        image = img_to_array(image)

    image = normalize(image)

    return image, steering_angle

#
# Define the Nvidia CNN model, 9 layers total
#   l1 - > Conv (24, 5, 5), Activaten: 'elu', MaxPooling=(2,2)
#   l2 - > Conv (36, 5, 5), Activaten: 'elu', MaxPooling=(2,2)
#   l3 - > Conv (48, 5, 5), Activaten: 'elu', MaxPooling=(2,2)
#   l4 - > Conv (64, 3, 3), Activaten: 'elu', MaxPooling
#   l5 - > Conv (64, 3, 3), Activaten: 'elu', MaxPooling, Flatten
#   l6 - > Dense (1164), Activation: 'elu', Dropout(0.5)
#   l7 - > Dense (100), Activation: 'elu', Dropout(0.5)
#   l8 - > Dense (50), Activation: 'elu', Dropout(0.5)
#   l9 - > Dense (10), Activation: 'elu', Dropout(0.5)
#   Output layer -> 1, Activation: 'linear'
def get_model(shape):

    model = Sequential()
    #model.add(BatchNormalization(mode=2, axis=1, input_shape=shape))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='elu',
                            input_shape=(shape[0], shape[1], 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

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

#
# Validation generator
def validation_generator(test_data, test_labels):
    while True:
        index = np.random.randint(len(test_data) - 1)
        images = test_data[index]
        angles = test_labels[index]

        index2 = np.random.randint(3)

        x = load_img(images[index2])
        x = preprocess_test_image(x)
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        y = np.array([[angles[index2]]])

        yield x, y


#
# Training generator
def training_generator(batch_size, train_data, train_labels):
    while True:
        x = []
        y = []

        for i in range(batch_size):
            index = np.random.randint(len(train_data) - 1)

            images = train_data[index]
            angles = train_labels[index]

            index2 = np.random.randint(3)
            image, angle = load_and_process_image(images[index2], angles[index2])

            x.append(image)
            y.append(angle)

        yield np.array(x), np.array(y)


#
# This methods loads, preprocess the data and trains the model.
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

    model.fit_generator(training_generator(64, features_train, labels_train),
                        samples_per_epoch=20480,
                        nb_epoch=2,
                        verbose=1,
                        callbacks=[],
                        validation_data=validation_generator(features_test, labels_test),
                        nb_val_samples=len(features_test))

    with open("model.json", "w") as fp:
        json.dump(model.to_json(), fp)

    model.save_weights("model.h5", overwrite=True)
    print("Done!")


#
# Main method
if __name__ == '__main__':
    train_model()