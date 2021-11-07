#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    train.py.py
# @Author:      Daniel Puente Ramírez
# @Time:        7/11/21 17:32


# Parameters
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob


def main():
    epochs = 100
    lr = 1e-4
    batch_size = 64
    img_dims = (96, 96, 3)

    data = []
    labels = []

    # Load image files from the dataset
    image_files = [f for f in glob.glob(r'faces_dataset' + "/**/*", recursive=True) if not os.path.isdir(f)]
    random.shuffle(image_files)

    # Convert images to arrays and labelling the categories
    for img in image_files:
        data.append(img_to_array(cv2.resize(cv2.imread(img), (img_dims[0], img_dims[1]))))

        label = 0 if img.split(os.path.sep)[-2] == 0 else 1  # if the label is men assign 0, if it is a woman 1
        label.append([label])

    # Starts the preprocessing
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # Split the dataset for training and validation
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Dataset augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    # Built the model
    model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                  classes=2)

    # Compile the model
    opt = Adam(lr=lr, decay=lr/epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the model
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY),
                            steps_per_epoch=len(trainX), epochs=epochs, verbose=1)

    # Save the model to disk
    model.save('gender_detection.model')

    # Plot training/validation loss/accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="best", frameon=True)

    # Save plot to disk
    plt.savefig('plot.png')


def build(width, height, depth, classes):
    """
    Model definition.
    :param width:
    :param height:
    :param depth:
    :param classes:
    :return:
    """
    model = Sequential()
    input_shape = (height, width, depth)
    chan_dim = -1

    if K.image_data_format() == "channels_first":  # Returns a string, channels_{first, last}
        input_shape = (depth, height, width)
        chan_dim = 1

    """ The axis that should be normalized, after a Conv2D layer with data_format="channels_first", set axis=1 in 
    BatchNormalization. """
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chan_dim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chan_dim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model


if __name__ == '__main__':
    main()
