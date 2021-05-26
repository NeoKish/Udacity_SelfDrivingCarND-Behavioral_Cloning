
# Importing relevant libraries
import csv
import cv2
import numpy as np
import sklearn
import math
from random import shuffle

# Extracting data from CSV line by line and storing into samples array
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
      samples.append(line)
#   This removes the first line of heading
    samples=samples[1:]    

#   Splitting the data into training and validation data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


#   Generator function to process the images in batches to avoid saving it once and overloading the memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                
                steering_center = float(batch_sample[3])

            # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                path = "./data/IMG/" # fill in the path to your training IMG directory
                img_center = cv2.imread(path+batch_sample[0].split('/')[-1])
                img_left = cv2.imread(path+batch_sample[1].split('/')[-1])
                img_right = cv2.imread(path+batch_sample[2].split('/')[-1])

                # add images and angles to data set
                car_images.append(img_center)
                car_images.append(img_left)
                car_images.append(img_right)
                steering_angles.append(steering_center)
                steering_angles.append(steering_left)
                steering_angles.append(steering_right)
                
                # add flipped images and flipped steering angles to avoid sticking to one turn                
                car_images.append(cv2.flip(img_center,1))
                car_images.append(cv2.flip(img_left,1))
                car_images.append(cv2.flip(img_right,1))
                steering_angles.append(steering_center*-1)
                steering_angles.append(steering_left*-1)
                steering_angles.append(steering_right*-1)

           
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Importing keras functions for model development,training and validating data
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda , Dropout ,Cropping2D
from keras.layers.convolutional import Conv2D

model=Sequential()
# Normalizing the image and cropping the image for better model development
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Developed the model using NVIDIA's end to end learning PilotNet architecture. Only addition is dropout method after first convolutional network
model.add(Conv2D(24,(5,5),subsample=(2,2),activation="relu"))
# Added dropout regularization of 50% .This helped in better generalization of the model without addition of extra data points forn sharp curve
model.add(Dropout(0.5))
model.add(Conv2D(36,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
# model.add(Dropout(0.75))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# We are using mean square error for loss function since it is a regression network and for optimizer we are using Adam optimizer
model.compile(loss='mse', optimizer='adam')

# Training the model for 5 epochs
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)
# model.fit(X_train, y_train,validation_split=0.2,shuffle=True, epochs=5)

# Saving the model
model.save('model.h5')


    
    
    
    