import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/2/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    images.append(image)
                correction = 0.2
                angle = float(batch_sample[3])

                angles.append(angle)
                angles.append(angle + correction)
                angles.append(angle - correction)

            yield shuffle(np.array(images), np.array(angles))

measurments = []

with open('./data/2/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        measurments.append(line)

train_samples, validation_samples = train_test_split(measurments, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

model = Sequential()

#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, trainable=False))

model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))

model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))

model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))

model.add(Conv2D(64,3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))

model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())

#model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3, validation_data=validation_generator, nb_val_samples=len(validation_samples)*3, nb_epoch=3)

model.save("model.h5")
