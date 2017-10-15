# **Traffic Sign Recognition** 


---

## Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg        "Grayscaling"
[image3]: ./examples/random_noise.jpg     "Random Noise"
[image4]: ./examples/20kmhr.jpg           "20 km/hr speed limit"
[image5]: ./examples/ahead_only.jpg       "Ahead only"
[image6]: ./examples/road_work.jpg        "Road work"
[image7]: ./examples/stop.jpg             "Stop sign"
[image8]: ./examples/turn_right_ahead.jpg "Turn right ahead"
[training_hist]: ./examples/training_hist.png
[valid_hist]: ./examples/validation_hist.png
[test_hist]: ./examples/test_hist.png
![20kmhr_orig_img][./examples/20kmhr_orig.png]
![20kmhr_mod_img][./examples/20kmhr_orig_mod.png]




## Rubric Points
Here I will provide a reference to the sections below that address each individual rubric. The rubric points and descriptions for this project may be found [here](https://review.udacity.com/#!/rubrics/481/view).

- Dataset Exploration
  - [Dataset Summary](#dataset-summary)
  - [Exploratory Visualization](#exploratory-visualization)
- [Design and Test a Model Architecture]
  - [Preprocessing] (#preprocessing)
  - [Model Architecture] (#model-architecture)
  - [Model Training] (#model-training)
  - [Solution Approach] (#solution-approach)
- [Test A Model On New Images]
  - [Acquiring New Images] (#acquiring-new-images)
  - [Performance on New Images] (#performance-on-new-images)
  - [Model Certainty Softmax Probabilities] (#model-certainty-softmax-probabilities)
 

## Data Set Summary & Exploration

## Dataset Summary

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is *34799*
* The size of the validation set is *4410*
* The size of test set is *12630*
* The shape of a traffic sign image is *32*x*32*x*3*, 32x32 pixels with 3 color channels  
* The number of unique classes/labels in the data set is *43*. This is the number of types of signs considered in these datasets.


### Exploratory Visualization

For each of the three data sets, we plot a histogram showing the distribution of sign types, and subsequently plot two examples of every sign type (see the notebook for these pictures). Note that the _relative_ distributions have roughly the same structure in each histogram. 

![Training][training_hist]
![Validation][valid_hist]
![Test][test_hist]



##  Design and Test a Model Architecture

### Preprocessing 

The first preprocessing step I considered was normalization. I tried

  i)normalizing each color channel for all data sets to zero mean and unit variance,
  ii) and normalizing according to pixel = (pixel - 128)/128.0.

Unfortunately this never seemed to produce a good result. The model always seemed to get trapped in a 
local minimum. I have included the code for the normalization in the notebook but left it 
commented out. It may be that normalization would improve results with the addition of other processing
steps I tried later (e.g. training set augmentation, drop out on the network). 

As I think is often the case in machine learning, more data will generally produce better (and less over-fit) results, 
than making very complex models. I therefore decided to augment the training data by generating 
random rotations and scalings of each image in the original set and adding back, thereby 
doubling its size. Each rotation and scaling was chosen as a random number from a uniform distribution between +/- 10 degrees and (0.85,1.25), respectively.  

Below is an example of the first 20 km/hr speed limit sign in the training set and the same sign
after a random rotation and scaling. 

![20kmhr_orig][20kmhr_orig_img]     ![20kmhr_mod][20kmhr_mod_img]


I began by considering modifications to the LeNet architecture without modifying the training data, specifically removing the pooling in one or both of the two CovNet layers and adding a third Covnet layer. I was unable to obtain anything better than about 85% validation accuracy. By adding dropout on the fully connected layers, I was able to increase this result to about 90%. 




####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Model training 


The basic Lenet architecture was used for the model, with dropout included on the 
fully connected layers. The Adam optimizer was used for computing and applying gradients. 
I found that using a slightly smaller batch size (100) than the Lenet architecture default 
(128), and slightly smaller learning rate produced the best results. It was necessary to 
increase the number of epochs from the default of 10 to 15, to ensure a validation 
accuracy that well clears the 93% threshold. Unfortunately, this does produce a 
bit of overfitting.  




####4. Model Results

Different model architectures and tuning parameters that were considered were discussed 
above. 


My final model results were:
* Training set accuracy of ?
* Validation set accuracy of ? 
* Test set accuracy of ?

 

###Test a Model on New Images

I found the five following German traffic signs on the web and provide them in the report

Here are five German traffic signs that I found on the web:

![20kmhr][./examples/20kmhr.jpg]
![Road Work][./examples/road_work.jpg]
![Ahead Only][./examples/ahead_only.jpg]
![Turn Right Ahead][./examples/turn_right_ahead.jpg]
![Stop][./examples/stop.jpg]



An "ahead only or turn right head" combination sign may be difficult to pick out from an
individual "ahead only" or "turn right head" sign for obvious reasons. The road_work sign
looks very similar to the signs. 

Interestingly, before arriving at the final parameters for the model architecture, the red 
and white Stop sign was often found to be classified as the Priority Road sign. Perhaps the 
model had a hard time distinguishing the nearly solid white in the center of the Stop sign 
(as the image is quite low resolution) and the yellow in the center of the Priority Road sign.
   

####2 Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).



Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (20km/h)      		| Speed Limit (20km/h)   									| 
| Road Work     			| Road Work										|
| Turn Right Ahead				| Turn Right Ahead 											|
| Ahead Only	      		| Ahead Only					 				|
| Stop Sign			| Priority Road       							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares somwhat favorably to the accuracy on the test set of 93%. 

####3. 


Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The top five soft max probabilities  for the 5 downloaded images were as follows:


20 km/h Speed Limit

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 20 km/h Speed Limit   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


Road Work 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 20 km/h Speed Limit   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


Ahead Only 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 20 km/h Speed Limit   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


Turn Right Ahead

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 20 km/h Speed Limit   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

Stop sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 20 km/h Speed Limit   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


