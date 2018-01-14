
import os
import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import sklearn
import timeit

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Conv2D, MaxPooling2D


cwd = os.getcwd()

# read images log file
lines = []

with open('data/driving_log.csv') as csvfle:
    reader = csv.reader(csvfle)
    for line in reader:
        lines.append(line)

header = lines[0]
del lines[0]

images = []
measurements = []  # list to hold all steering angles
correction = 0.25  # steering correction for the left and right camera


for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = cwd + '/data/IMG/' + filename
        image = mpimg.imread(current_path)
        images.append(image)
    # implement steering correction for the left and right camera, save steering with corrections
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)

augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

samples = lines
ch, row, col = 3, 80, 320  # Trimmed image format
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        samples = np.random.permutation(samples)
        for offset in range(0, num_samples - batch_size, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = mpimg.imread(cwd + '/data/IMG/' + batch_sample[0].split('/')[-1])
                left_image = mpimg.imread(cwd + '/data/IMG/' + batch_sample[1].split('/')[-1])
                right_image = mpimg.imread(cwd + '/data/IMG/' + batch_sample[2].split('/')[-1])

                images.append(center_image)
                images.append(left_image)
                images.append(right_image)

                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# modified NVDIA architecture
model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = (160, 320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(24,(5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(36,(5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(48,(5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


start_time = timeit.default_timer()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
                                     validation_data=validation_generator, nb_val_samples=len(validation_samples),
                                     nb_epoch=4)
model.save('model.h5')
elapsed = timeit.default_timer() - start_time
print('time to train modified NVDIA model using generator: %.2f' % (elapsed))


model.save_weights('model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)