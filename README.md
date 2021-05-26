# **Behavioral Cloning** 

This project is part of Udacity's Self- Driving Car Nanodegree Course. In this section of the course, we were introduced to Keras framework and Transfer Learning. The goal of the project is to build a Neural Network model which would be able to autonomously drive car in a simulator track provided by Udacity by using the training data obtained by collecting camera image data from manually driven car in the same track.

The input for our model training is camera images obtained from car and output is the steering angle which helps in keeping the car on the road section during autonomous mode without driving into non-road parts such as trees, curb or water.


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (File References)

[Video]: ./video.mp4
[data]:  ./data/driving_log.csv
[image_data]: ./data/IMG    
[flipped_image_data]: ./data/IMG



As part of the submission, the project includes the following files

#### 1. Project Files
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* [Video] file reconstructed from the images while running the car in autonomous mode

#### 2. Submission 

The model.py includes code for retrieving images from [data] file and storing them into [image_data] file. Images from [image_data] were separated out into left, center and right and flipped to improve the model. The code has been properly commented with reasons for reader usability.

The code has been executed and the model has been validated and saved in model.h5 file

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture

1. Data 
Image data stored in [image_data] are separated out into left, center and right camera images. Corresponding to this images, steering angles are stored into sample set. For left and right steering a correction factor is added to steering angle value as this is not something obtained from the dataset. During first few runs of the model it was observed that car was steering to left a lot which could be explained due to more of left turns in the simulator track. To mitigate this problem, flipped images for all left, right and center images were added to the sample data with negative steering angles. Using sklearn train_test_split function, training data and validation data were created with 20 percent split.

2. Generator function
Due to limited memory available on the workspace, a generator function was used to process and train images in batches using keras model.fit_generator function.

3. Preprocessing
Before the images are fed into the model, they are normalized and are cropped to appropriate size to eliminated non-road features which could cause model to train differently

3. Model Architecture
NVIDIA's end to end learning PilotNet (https://developer.nvidia.com/blog/explaining-deep-learning-self-driving-car/) architecture has been used which involves 5 convolutional layers ,3 fully connected layers and output layer giving steering control values. First three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.To introduce non-linearity in convolutional layers, activation function ReLU as it helps in training the model better.

Mean square error is used for loss function instead of cross entropy since it is a regression network and not classification network and for optimizer we are using Adam optimizer.

The model architecture works well for most part of autonomous driving but has issues when the car comes across sharp curves. The problem could have been resolved by adding more training data with sharp curves but instead adding dropout function after first convolutional layer helped in generalising the model much better. 








