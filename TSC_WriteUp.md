# **Traffic Sign Recognition** 


---

## Build a Traffic Sign Recognition Project 

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg        "Grayscaling"
[image3]: ./examples/random_noise.jpg     "Random Noise"
[20kmhr_img]: ./examples/20kmhr.jpg           "20 km/hr speed limit"
[ahead_only_img]: ./examples/ahead_only.jpg       "Ahead only"
[road_work_img]: ./examples/road_work.jpg        "Road work"
[stop_img]: ./examples/stop.jpg             "Stop sign"
[turn_right_ahead_img]: ./examples/turn_right_ahead.jpg  "Turn right ahead"
[training_hist]: ./examples/training_hist.png
[valid_hist]: ./examples/validation_hist.png
[test_hist]: ./examples/test_hist.png
[20kmhr_orig_img]: ./examples/20kmhr_from_train.png
[20kmhr_mod_img]: ./examples/20kmhr_from_train_mod.png




## Rubric Points
Here I will provide a reference to the sections below that address each individual rubric. The rubric points and descriptions for this project may be found [here](https://review.udacity.com/#!/rubrics/481/view).

- Dataset Exploration
  - [Dataset Summary](#dataset-summary)
  - [Exploratory Visualization](#exploratory-visualization)
- Design and Test a Model Architecture
  - [Preprocessing](#preprocessing)
  - [Model Architecture](#model-architecture)
  - [Model Training](#model-training)
  - [Solution Approach](#solution-approach)
- Test A Model On New Images
  - [Acquiring New Images](#acquiring-new-images)
  - [Performance on New Images](#performance-on-new-images)
  - [Model Certainty Softmax Probabilities](#model-certainty-softmax-probabilities)
 

## Data Set Summary & Exploration

### Dataset Summary

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of the original training set is *34799*
* The size of the validation set is *4410*
* The size of test set is *12630*
* The shape of a traffic sign image is *32*x*32*x*3*, 32x32 pixels with 3 color channels  
* The number of unique classes/labels in the data set is *43*. This is the number of types of signs considered in these datasets.


### Exploratory Visualization

For each of the three data sets, we plot a histogram showing the distribution of sign types, and subsequently plot two examples of every sign type (see the notebook for these pictures). The file signnames.csv contains the mapping from sign type to index. Note that the _relative_ distributions have roughly the same structure in each histogram. 

![Training][training_hist]
![Validation][valid_hist]
![Test][test_hist]



##  Design and Test a Model Architecture

### Preprocessing 

The first preprocessing step I considered was normalization. I tried

i)normalizing each color channel for all data sets to zero mean and unit variance,

ii) and normalizing according to pixel = (pixel - 128)/128.0.

Unfortunately neither of these ever seemed to produce a good result. The model always seemed to get trapped in a 
local minimum, never getting above about 83% vaildation accuracy. I have included the code for the normalization in
the notebook but left it commented out. It may be that normalization would improve results with the addition of further processing steps beyond that which I implemented (e.g. training set augmentation, drop out on the network). 

As I think is often the case in machine learning, more data will generally produce better (and less over-fit) results, 
than making very complex models. I therefore decided to augment the training data by generating 
random rotations and scalings of each image in the original set, twice, and then added back, thereby 
tripling the original size of the training set. Each rotation and scaling was chosen as a random number from a uniform distribution between +/- 10 degrees and (0.85,1.25), respectively.  

Below is an example of the first 20 km/hr speed limit sign in the training set and the same sign
after a random rotation and scaling. 

![20kmhr_orig][20kmhr_orig_img]     ![20kmhr_mod][20kmhr_mod_img]



### Model Architecture

I began by considering modifications to the LeNet architecture without modifying the training data, specifically removing the pooling in one or both of the two convolutional layers and adding a third convolutional layer. I was unable to obtain anything better than about 85% validation accuracy. By adding dropout on the fully connected layers, I was able to increase this result to about 90%. With addition of the augmented data as described above, I was able to achieve 94-95% validation accuracy.

My final model is essentially the same as the LeNet architecture, consisting of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution       | 5x5 filter, 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					    |		outputs 28x28x6										|
| Max pooling	      | 2x2 stride,  outputs 14x14x6 				|
| Convolution 	    |  5x5 filter, 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					    |		outputs 14x14x16										|
| Max pooling	      | 2x2 stride,  outputs 5x5x16 (=400)		|
| Fully connected		|   400x120     	|
| RELU					    |		outputs 120			|
| Dropout           | keep probability 70%  |
| Fully connected		|   120x84     	|
| RELU					    |		outputs 84			|
| Dropout           | keep probability 70%  |
| Fully connected		|   84x43     	|
| RELU					    |		outputs 43 (# of sign types)		|


### Model Training 

The basic Lenet architecture was used for the model, with dropout included on the 
fully connected layers. The Adam optimizer was used for computing and applying gradients. 
I found that using a slightly smaller batch size (100) than the Lenet architecture default 
(128), and slightly smaller learning rate produced the best results. It was found that 7-10 epochs 
was necessary to get a validation accuracy that well clears the 93% threshold. 


### Solution Approach

Different model architectures and tuning parameters that were considered to develop the final solution were discussed 
above. 

My final model results were:
* Training set accuracy of 98.5%
* Validation set accuracy of 94.5% 
* Test set accuracy of 94.0%

 

## Test a Model on New Images

The following is a discussion of running the the validated model against five images of German traffic signs
found on the web.

### Acquiring New Images
I found the five following German traffic signs on the web:


![20kmhr][20kmhr_img]
![Road Work][road_work_img]
![Ahead Only][ahead_only_img]
![Turn Right Ahead][turn_right_ahead_img]
![Stop][stop_img]


An "ahead only or turn right head" combination sign may be difficult to pick out from an
individual "ahead only" or "turn right head" sign for obvious reasons. Also they look somewhat similar to each other, which is why I chose them for the test here.

Interestingly, before arriving at the final parameters for the model architecture, the turn right 
ahead sign was often classified incorrectly, perhaps because it appears somewhat small (not well cropped)
in the overall picture. The Stop sign was also sometimes found to be classified as the Priority Road sign. Perhaps the 
model had a hard time distinguishing the nearly solid white in the center of the Stop sign 
(as the image is quite low resolution) and the yellow in the center of the Priority Road sign.
   

### Performance on New Images

Here are the model predictions running against the above five images:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (20km/h)      		| Speed Limit (20km/h)   									| 
| Road Work     			| Bicycles crossing 										|
| Turn Right Ahead				| Turn Right Ahead 											|
| Ahead Only	      		| Ahead Only					 				|
| Stop Sign			| Stop Sign       							|


The final model was able to correctly identify  4 out of the 5 traffic signs, which gives an accuracy of 80%.   This result compares favorably to the accuracy on the test set of 93.3%.  One can see below that the incorrect prediction, that for the Road Work Sign, has the least certainty in comparison to the other signs, 45% (Bicycle crossing) in comparison to essentially 100% for the others. The model does at least give 5% probability that the sign is indeed a Road Work sign.


### Model Certainty Softmax Probabilities

The certainty of the model is given by the softmax probabilities for each prediction.
The code for making predictions on the final model is located in the Ipython notebook cell under the heading
**Output Top 5 Softmax Probabilities For Each Image Found on the Web**.

The top five soft max probabilities  for the 5 downloaded images were as follows:




**20 km/h Speed Limit**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| 20 km/h Speed Limit   									| 
| .000     				  | 30 km/h Speed Limit									|
| .000				      | 70 km/h Speed Limit											|
| .000	      			| 120 km/h Speed Limit					 				|
| .000				      | 100 km/h Speed Limit      							|


**Road Work** 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .451         			| Bicycles crossing 									| 
| .317    				  | Road Narrows on the right 										|
| .102				      | Double Curve										|
| .005	      			| Beware of ice/snow					 				|
| .005				      | Road Work     							|


**Turn Right Ahead** 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			    | Turn Right Ahead   									| 
| .000          				| Roundabout mandatory										|
| .000				          | Ahead only											|
| .000	      			    | Keep Left 					 				|
| .000				          | Vehicles over 3.5 metric tons prohibited      							|


**Ahead Only**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			    | Ahead Only   									| 
| .000     				      | Turn Right Ahead 										|
| .000					          | Turn Left Ahead											|
| .000      			      | 	Roundabout mandatory				 				|
| .000			            | Keep Right   							|

**Stop sign**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop sign   									| 
| .000     				  | Road Work										|
| .000					| No entry 										|
| .000	      			| Yield				 				|
| .000				    | Speed limit (30km/h)      							|


