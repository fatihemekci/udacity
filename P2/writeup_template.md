# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./input_signs.png "Input Signs"
[image2]: ./sample_test.png "Sample Test"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is my python notebook as html attached.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Random samples from input images can be found below.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to zero mean unit variance to supress RGB values. This is only pre-processing I have applied. It already boost my accuracy from 84% to 90+% I created copy of images in order to normalize.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valie padding, outputs 32x32x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		|  300  |
| Fully connected		|  200  |
| Fully connected		|  43  |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used learning rate as 0.001. Changing learn rate did not increase my accuracy. 10 epochs was enough. I saw accuracy starts to fluctuate around 8th epoch. That is why I did not go any higher. Hyperparameters did not affect my architectures performance. I kept it as 0 mean 0.1 variance.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.938
* validation set accuracy of 0.938
* test set accuracy of 0.921

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
	* I started with a one layer NN, to understand input images. It did not give me good results but give good insights about input dataset.
* What were some problems with the initial architecture?
	* One layer was not enough to identify objects. It got stuck at lines level.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	* I changed my architecture to 5 layers since it is proven a good architecture. This gave me good accuracy on both validation and testing. This shows it is fitting well neither overfitting nor underfitting.
* Which parameters were tuned? How were they adjusted and why?
	* Learning rate, mu and sigma are adjusted to give better accuracy rate.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	* Model should be location invariant. Signs can be at any location of an image. Thus, convolutional layer is a must in this problem. Number of convolutional layers can be changed based on computation power. I stopped at 2 convolutional layer.

If a well known architecture was chosen:
* What architecture was chosen?
	* LeNet architecture is a good fit. We can consider traffic signs as alphabet.
* Why did you believe it would be relevant to the traffic sign application?
	* Classifying letters and traffic sign letters are very similar. We have limited number of signs and several sample images as in LeNet character recognition.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	* Both validation and test accuracy are very close. This indicates model fit well to our dataset.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I used following 10 images from test set.

![alt text][image2]

I think these images are good enough, some images are dark, some are light, some are blocked by other objects.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test/validation set. Model can not identify one sign that is covered by an object. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Model is relatively sure about its prediction for all images. Softmax value is greater than 0.86 (mostly 1.0) for all images except misclassified one. Misclassified image's softmax value is 0.59. This explains its poor performance. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


