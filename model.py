import csv
import cv2
import numpy as np

images = []
measurments = []
correction = 0.2
drop_out_rate = 0.5

with open("./data/driving_log.csv") as f:
    reader = csv.reader(f)

    for line in reader:
        angle_center = float(line[3])
        angle_right = angle_center - correction
        angle_left = angle_center + correction

        for i in range(3):
            image_name = line[i].split('/')[-1]
            images.append(cv2.imread("./data/IMG/" + image_name))

        measurments.append(angle_center)
        measurments.append(angle_left)
        measurments.append(angle_right)

x_train = np.array(images)
y_train = np.array(measurments)

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D,Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))

model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120))
model.add(Dropout(drop_out_rate))

model.add(Dense(84))
model.add(Dropout(drop_out_rate))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=5)
model.save("model.h5")
