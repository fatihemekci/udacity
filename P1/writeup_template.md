# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline and draw_lines (Prune lines)
Basic pipeline has 5 stes:
* Convert image to gray scale
* Gaussian smoothing
* Detect edges
* Hough transform
* Draw lines

Above steps are enough on a high level basis. However, it is not elegant enough and very fragile. Improved pipeline steps are:
* Get interested area: This is a bigger area than the original interested area. This will reduce area of interest and get rid of several noisy edges.
* Convert image to gray scale: Basic thresholding.
* Gaussian smoothing: Remove some noise.
* Detect edges using Canny edge detector.
* Get real interested area: This is area that lines can be really located.
* Detect lines by using hough transform with proper parameters.
* Prune lines: Lane lines should follow similar slopes, one set with positive slope and another set with negative slope. Also, slope shoud be in predefined range (absolute value of slope should be greater than 0.4 based on tests). Only select lines with predefined slope range. Also, calculate average slope for left and right lanes (negative slopes vs positive slopes.
* Interpolate lines: Find bottom and top limits of lines and interpolate lines based on average slopes.
* Draw just 2 lines one for right lane one for left lane.

### 2. Identify potential shortcomings with your current pipeline

* Current pipeline does not handle curves gracefully since lines are defined based on slopes.
* Area of interest selection can not handle uphill/downhill conditions since it is just based on image height and width.
* There are some hardcoded thresholds.
* Video processing does not take previous frames lane information to predict current frame's lane information.

### 3. Suggest possible improvements to your pipeline

* Curves: Lanes are not straight at curves and defining lines with slopes can not handle curves. Instead, lanes should be defined as a part of ellipse. By changing radiuses and part of ellipse, lanes can be defined for both curves and straigh lanes.
* Thresholding is not best way for anything: All should be adaptive.
** Different lighting condition: Thresholding is not a good option considering lighting conditions can change like day/night lights.
** Area of interest selection: Area of interest should be from bottom of image to horizon. However, horizon detection is not easy at road images.
* Consequtive frames should have lanes at very close locations. Instead of focusing to current frame, lane information from previous images should be aggregated and updated based on current frame's data. 



