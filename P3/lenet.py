import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
images = []
measurements = []
sample_cnt = 0
with open('..\..\data\data\driving_log.csv') as  in_file:
    reader = csv.reader(in_file)
    for line in reader:
        # Sample original data. They are all coming from car driving at center of the road.
        if 'udacity' in line[0] and np.random.rand() > 0.75:
            continue
        if 'udacity' not in line[0] and np.random.rand() > 0.95:
            continue
        lines.append(line)
        filename_prefix = '..\..\data\data\\'
        if 'udacity' in line[0]:
            filename_prefix = ''
        filename = filename_prefix + line[0]
        image = cv2.imread(filename)
        measurement = float(line[3])
        # Add center image
        images.append(image)
        measurements.append(measurement)
        # Add inverted center image
        images.append(np.fliplr(image))
        measurements.append(-1 * measurement)
        # Add left image
        #filename = filename_prefix + line[1].replace(' ', '')
        #image = cv2.imread(filename)
        #images.append(image)
        #measurements.append(measurement + 0.2)
        # Add right image
        #filename = filename_prefix + line[2].replace(' ', '')
        #image = cv2.imread(filename)
        #images.append(image)
        #measurements.append(measurement - 0.20)

X_train = np.array(images)
Y_train = np.array(measurements)

print(X_train.shape)
print(Y_train.shape)

model = Sequential()
# Pre-processing: Crop images and normalize
model.add(Lambda(lambda x : (x / 127.5) - 1.0, input_shape=X_train.shape[1:]))
model.add(Cropping2D(cropping=((50, 20), (0,0))))

# First convolutional layer
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))

# Second convolutional layer
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.5))

# Flatten and reduce dimension
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# Compile and fit model
model.compile(loss='mse', optimizer='adam')
# I can handle loading all data to memory. No need to fit_generator.
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('lenet.h5')
