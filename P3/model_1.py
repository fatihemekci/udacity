import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

lines = []
images = []
measurements = []
sample_cnt = 0
with open('..\..\data\data\driving_log.csv') as  in_file:
    reader = csv.reader(in_file)
    for line in reader:
        lines.append(line)
        image = cv2.imread('..\..\data\data\\' + line[0])
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        sample_cnt += 1
        if sample_cnt > 10000:
            break

X_train = np.array(images)
Y_train = np.array(measurements)

print(X_train.shape)
print(Y_train.shape)


model = Sequential()
model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=X_train.shape[1:]))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model_1.h5')
