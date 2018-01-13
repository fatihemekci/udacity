import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import sklearn
from sklearn.model_selection import train_test_split


def generator(samples, batch_size=1024):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for imagePath, measurement in batch_samples:
                if np.random.rand() > 1.95:
                    print("{}@@{}".format(imagePath, measurement))
                image_orig = cv2.imread(imagePath)
                image = cv2.GaussianBlur(image_orig, (3,3), 0)
                images.append(image)
                measurements.append(measurement)
                images.append(np.flip(image, 1))
                measurements.append(measurement*-1.0)

            yield sklearn.utils.shuffle(np.array(images), np.array(measurements))


def read_input():
    files = []
    measurements = []
    with open('..\..\data\data\driving_log.csv') as  in_file:
        reader = csv.reader(in_file)
        for line in reader:
            # Sample original data. They are all coming from car driving at center of the road.
            if 'udacity' in line[0] and np.random.rand() > 1.75:
                continue
            if 'udacity' not in line[0] and np.random.rand() > 1.095:
                continue
            filename_prefix = '..\..\data\data\\'
            if 'udacity' in line[0]:
                filename_prefix = ''
            filename = filename_prefix + line[0]
            measurement = float(line[3])
            files.append(filename)
            measurements.append(measurement)
            # Add left image
            filename = filename_prefix + line[1].replace(' ', '')
            files.append(filename)
            measurements.append(measurement + 0.2)
            # Add right image
            filename = filename_prefix + line[2].replace(' ', '')
            files.append(filename)
            measurements.append(measurement - 0.2)
    return (files, measurements)
    
def create_model():
    model = Sequential()
    # Pre-processing: Crop images and normalize
    model.add(Lambda(lambda x : (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0,0))))

    # First convolutional layer
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.5))

    # Second convolutional layer
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.5))

    # Third convolutional layer
    model.add(Convolution2D(48, 5, 5, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    # Fourth convolutional layer
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    # Fifth convolutional layer
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    # Flatten and reduce dimension
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

files, measurements = read_input()
samples = list(zip(files, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

model = create_model()


# Compile and fit model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=2, verbose=1)

model.save('vgg.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
