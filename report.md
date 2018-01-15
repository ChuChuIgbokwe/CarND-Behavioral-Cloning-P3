# **Behavioral Cloning** 

## Report


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
* 
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_modified_NVDIA_generator.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

* The overall strategy for deriving a model architecture was to use what already exists. I chose the convolution neural network model similar to the [Nvidia model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
* I decided to do minimal image augmentation and use the raw data to establish a baseline performance.
* The final step was to run the simulator to see how well the car was driving around track one. It performed well on the training track, driving autonomously without ever leaving the road. The video of it can be found [here](https://www.youtube.com/watch?v=tHjLsKaeP_g&feature=youtu.be)


#### 2. Final Model Architecture

The final model architecture (model.py lines 105-119) consisted of a convolution neural network with the following layers 
```
sizes model = Sequential()
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
```

* The model includes RELU layers to introduce nonlinearity
* The data is normalized using a Keras lambda layer
* The model has 5 convolutional layers with 3x3 and 5x5 filter sizes and depths between 24 and 64
* The model has a dropout with a 0.5 probability rate to handle overfitting
* An Adam optimizer was used for optimization. This requires little or no tunning as the learning rate is adaptive


#### 3. Creation of the Training Set & Training Process

* To capture good driving behavior, I recorder four laps of with a logitech controller. Two laps driving clockwise and two laps driving counter clockwise. 
* I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
* I made no recovery runs and used the data as it was
* To augment the data set I added a correction value of 0.25 to images from the left camera nd subtracted 0.25 from images from the right camera.
* After the collection process, I had 21738 number of data points. I used 80% for training and 20% for validation.
* I passed the data into a generator and shuffled the output to be fed into the neural network and trained it over 4 epochs


####

#### Conclusion
I was able to get it working succesfully. Here are some of my thoughts regarding the project

1. Lack of image processing did not appear to affect the ability of the car to drive round the track successfully. However it meant that the car could not drive well on the challenge track.
2. I would like to extend the model to be able to handle the challenge track and also handle throttling and braking as I was driving at 31mph in training and I couldn't get above 10mph in testing
3. I would like to create a model that uses a singl lap and data augmentation to make the car able to drive in any kind of track.
4. Training the model on the challenge track and trying it out on the baseline track without any image processing is an excercise I'd liek to pursue.


#### References
1. https://github.com/rjmoss/CarND-Behavioural-Cloning-P3
