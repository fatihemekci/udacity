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
        sample_cnt += 1
        if sample_cnt % 2 == 0:
            continue
        if sample_cnt > 1200:
            break
        lines.append(line)
        image = cv2.imread('..\..\data\data\\' + line[0])
        measurement = float(line[3])

        images.append(image)
        measurements.append(measurement)

        images.append(np.fliplr(image))
        measurements.append(-1 * measurement)
        print("{} {}".format(sample_cnt, len(images)))

X_train = np.array(images)
Y_train = np.array(measurements)

print(X_train.shape)
print(Y_train.shape)


model = Sequential()
model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=X_train.shape[1:]))
model.add(Cropping2D(cropping=((50, 20), (0,0))))
model.add(Convolution2D(6, 5, 5))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(6, 5, 5))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('lenet.h5')
