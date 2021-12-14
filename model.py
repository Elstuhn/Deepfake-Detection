import tensorflow as tf
import numpy as np
from numpy import ndarray 
import pandas as pd
from tensorflow.keras.datasets import mnist
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, Dropout
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image
import os
from time import sleep
from typing import Tuple, List
import pickle

earlystop = EarlyStopping(monitor = "val_accuracy", patience = 3)
model_save = ModelCheckpoint('BloodCells.hdf5',
                            save_best_only=True)

def getmodel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (480, 480, 3)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
model = getmodel()
with open('dataset/trainX', 'rb') as readfile:
    trainX = pickle.load(readfile)

with open('dataset/testX', 'rb') as readfile:
    testX = pickle.load(readfile)

with open('dataset/trainY', 'rb') as readfile:
    trainY = pickle.load(readfile)
    
with open('dataset/testY', 'rb') as readfile:
    testY = pickle.load(readfile)
    
history = model.fit(trainX, trainY, validation_data = (testX, testY), callbacks = [earlystop, model_save])
score = model.evaluate(testX, testY, verbose = 0)

print("Test loss:", score[0])
print("Test accuracy:", score[1])

def plothist(history, metric):
    plt.figure()
    if metric == "accuracy":
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel("Accuracy")
    elif metric == "loss":
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"])
    plt.show()
    
plothist(history, "accuracy")
plothist(history, "loss")
