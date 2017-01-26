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


#
# load only 500 datasets with 0 steering
num_load_0_steering = 50
img_data   = []
angle_data = []
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
# normalize the image data
def normalize(X):
    X = X.astype('float32')
    X = X/255.0 - 0.5
    return X

def resize(img, height, length):
    return cv2.resize(img, (length, height), interpolation=cv2.INTER_CUBIC)

#
# pre-process the loaded image
def preprocess_image(img):
    img = crop(img)
    #img = rgb2hsv(img)
    img = resize(img, 128, 128)
    img = normalize(img)
    return img

def generate_training_data():
    steering_0 = 0
    csv = pd.read_csv(os.path.join(os.getcwd(), 'data', 'driving_log.csv'))

    for i, row in csv.iterrows():
        random_num = np.random.randint(3)
        steering_angle = row['steering']

        if random_num == 0:
            # choose the center image
            path = os.path.join(os.getcwd(), 'data', row['center'].strip())
        elif random_num == 1:
            # choose the left image
            path = os.path.join(os.getcwd(), 'data', row['left'].strip())
            steering_angle = steering_angle + 0.25
        else:
            # choose the right image
            path = os.path.join(os.getcwd(), 'data', row['right'].strip())
            steering_angle = steering_angle - 0.25

        img = load_image(path)
        img = preprocess_image(img)


        if abs(steering_angle) < 0.1:
            steering_0 = steering_0 + 1

        if steering_0 > num_load_0_steering:
            continue

        if abs(steering_angle) > 0.3:
            h_img = cv2.flip(img, 1)
            img_data.append(h_img)
            angle_data.append(-steering_angle)

        img_data.append(img)
        angle_data.append(steering_angle)



def get_model():
    model = Sequential()

    #
    # Modeling NVIDIA network here, add Batchnormalization
    model.add(BatchNormalization(mode=2, axis=1, input_shape=(128, 128, 3)))

    #
    # Convolution layer 1
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #
    # Convolution layer 2
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #
    # Convolution layer 3
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #
    # Convolution layer 4
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    #
    # Convolution layer 5
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    #
    # Flatten
    model.add(Flatten())

    #
    # Dense layer 1
    model.add(Dense(1164, activation='relu'))

    #
    # add some dropout layer to avoid overfitting
    model.add(Dropout(0.2))

    #
    # Dense layer 2
    model.add(Dense(100, activation='relu'))

    #
    # add some dropout layer to avoid overfitting
    model.add(Dropout(0.2))

    #
    # Dense layer 3
    model.add(Dense(50, activation='relu'))

    #
    # add some dropout layer to avoid overfitting
    model.add(Dropout(0.2))

    #
    # Dense layer 4
    model.add(Dense(10, activation='relu'))

    #
    # add some dropout layer to avoid overfitting
    model.add(Dropout(0.2))

    #
    # Output layer
    model.add(Dense(1, activation='relu'))

    return model


def training_generator(num_batches, batch_size):

    while True:
        x = np.zeros((batch_size, 128, 128, 3))
        y = np.zeros(batch_size)

        for n in range(num_batches):
            begin = (n * batch_size)
            end   = (n * batch_size) + batch_size

            x = img_data[begin:end]
            y = angle_data[begin:end]

            yield x, y

def main():

    #
    # Number of epochs
    nb_epoch = 5

    #
    # batch size
    n_batch_size = 64

    #
    # Generate training data
    generate_training_data()
    print("Data generate done")
    #num_batches = int(math.ceil(len(steering_data) / float(n_batch_size)))
    num_batches = int(len(steering_data) / float(n_batch_size))

    #
    # Get model & compile
    model = get_model()
    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])

    print("Model compiled")

    model.fit_generator(training_generator(num_batches, n_batch_size),
                        samples_per_epoch=num_batches * n_batch_size,
                        nb_epoch=1,
                        verbose=1,
                        callbacks=[],
                        validation_data=None)

    with open("model.json", "w") as fp:
        json.dump(model.to_json(), fp)

    model.save_weights("model.h5", overwrite=True)



if __name__ == '__main__':
    main()


