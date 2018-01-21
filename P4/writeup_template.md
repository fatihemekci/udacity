## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/sobel_binary_image.png "Sobel Result"
[image3]: ./output_images/sobel_then_perspective.png "Sobel Result With Perspective"
[image4]: ./output_images/perspective_transform_output.png "Perspective Transform Output"
[image5]: ./output_images/lane_fit.png "Sliding Window Fitted Lane"
[image6]: ./output_images/lane_with_text.png "Lanes With Info"
[image7]: ./output_images/lanes.png "Lanes"
[image8]: ./output_images/undistort_output_2.png "Undistorted"
[video1]: ./project_video_output.mp4 "Video"
[video2]: ./harder_challange_video_output.mp4 "Challenge video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in DistortionCorrector class. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image8]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once distortin matrices are extracted from sample images, images can be transformed by `cv2.undistort`. Sample undistortion is given below.

![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of Sobel magnitude and Sobel directions to generate binary images. I have used S channel from HLS channels. S channel is independent of lighting and good to use at different lighting conditions. Sobel angles is useful since lane lines angle should be in a defined range. Sobel magnitude is useful since there is always a color change between road and lanes. Here's an example of my output for this step.

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform (PerspectiveTransformer class) includes 2 main functions. First function is `train()` to extract transformation matrix. Second group of functions is `tranform()` functions to transform a given image based on previously extracted matrices. 
I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[490, 482],[810, 482],
                  [1250, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0], 
                  [1250, 720],[40, 720]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to detect lanes:
* Binarize the image
* Create a histogram counting white pixels at each x bin
* Find local maximax at left and right half of image
* Consider these local maximax as candidate regions and fit a polynomial by using a sliding window. 
* If lane is detected at previous frame, use previous image's lane location for starting point for search.

`findLines` and `findLinesFromPrevFit` are responsible from fitting a polynomial to detected lines. 

![alt text][image5]
![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Once lanes are detected and a polynomial is fitted on lanes, curvature can be calucated by using first and second derivates. Formula is already given at lecture notes. 

Since camera is mounted at the center of vehicle, vehicle center should be the center of image. By using fitted polynomials, center of lane can be calcualted at maximum y index. Difference between center of image (which is vehicle position) and center of lane is considered as location of vehicle with respect to vehicl.

I did this in at `singleImagePipeline` function. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I did this in at `singleImagePipeline` function. 

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [video1](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline depends on some thresholds. Thresholding can not be very robus given different road and lighting conditions. Unfortunately, my pipeline can not detect lanes on challenged video [video2] as good as it did on first video [video2]. Sobel can not detect binary image well that is why my pipeline can not find lanes. 

Detection based on previously detected lanes does not good results all time. Also, if there is some error on previosly detect lane, error cumulates over time causing more problems. Instead, I forced my pipeline to run a full search every at most 60 consecutive successful run.