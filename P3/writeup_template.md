# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used network architecture proposed by Nvidia. Architecture has 5 convolution layers and 4 fully connected layers. I have also tested LeNet architecture but it does not give good results.

#### 2. Attempts to reduce overfitting in the model

I have tested dropout layers but it did not improve model's performance. Also, I trained my model with 2 epochs. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I have used all images from center, left and right cameras. However, provided dataset was not enough to complete the track. Car was always pulling left and it could not complete curves successfully. I have added 3 more laps, 2 driving forward and 1 driving backward. Backward samples help the model to train right turns.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate. However, LeNet depth was not enough to learn tracks. I decided to use nVidia's architecture which has more convolutional layers and proven for self-driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Both training and validation sets has low mean squared errors. That is why I did not fight with overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I added more samples from those points. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes
* Lambda for normalization
* Cropping to remove unnecessary parts of images
* Conv (24, 5, 5)
* Conv (36, 5, 5)
* Conv (48, 5, 5)
* Conv (64, 3, 3)
* Conv (64, 3, 3)
* Dense layers

#### 3. Creation of the Training Set & Training Process

I used all images from orginal dataset. But model wont be able to recover from curves and was pulling left. To fix these issues, I collect more samples from track both driving forward and backward. Then, split the dataset to training and validation by 80 to 20%/

I used gaussian blur to remove noises. Also, updated drive.py to send blured image for prediction.

I finally randomly shuffled the data set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I have tested several epochs but epochs greater than 2 did not improve performance much. I used an adam optimizer so that manually training the learning rate wasn't necessary.
