


### Prediction using Human Activity Recognition Data

#### Executive Summary

In this study we use data from a Human Activity Recognition Experiment[1] to
create a prediction model. This model is then used to predict a test sample
of 20 observations. The raw data consists of about 160 variables. Using
various techniques this is reduced to about 40. We tried fitting Recursive
Partitioning and Random Forest models on the data, and based on the output
decided to select Random Forest as the final model. Then we tried fitting
Random Forest with an increasing number of variables until the out of
sample error rate reduced to below 1%. This was then used to predict the
outcome from the test sample.

**Code :** This report was generated from a Rmd file. For the sake of
clarity the code has been omitted here. The complete code is available from
the repository at [https://github.com/ajtm/CourseraPML](https://github.com/ajtm/CourseraPML)

#### Exploring and Preparing the Data

The data from the HAR activity [1] was provided as a part of an assignment.
This data was collected from subjects using wearable accelerometers and
peforming **Unilateral Dumbbell Biceps Curls**. The intention of the study is
to predict - based on the data - whether the exercises are being done correctly.
One of the variables
is *classe* which has five levels. Level-*A* represents an exercise done
correctly, while Level-*B* through Level-*E* represent different types of
mistakes. The data consists of 19622 observations on 160 variables.



We start by examining the data for NAs. While reading in the data we specified
both blank elements and NAs to be treated as NAs.
We find that 100 variables have 19216 NAs each.
This corresponds to
97.9 % missing
values for these variables.  As
this percentage of missing observations is very high and cannot be imputed,
we drop these columns. This leaves us with 60 useable variables including
the outcome.



**Cross Validation**  
For this study we plan to use Cross-Validation for evaluating our model. So,
the next step is to partition the data into training and testing sets. The
training set will contain 75% of the observations, randomly selected, and
the remaining 25% will be the test set. In order to avoid confusion, we will
refer to this test set as the cv-test set.
We have another dataset, consisting of 20 observations on which the final
prediction has to be made. We will call this the final-test set.



Our training dataset now contains 14718 observations of 60
variables.

#### Reducing Variables

At this point we still have too many variables. As a first step we will use
Principal Component Analysis to reduce some of the variables.
PCA will be done only on those variables that
have a high correlation, as that is where it gives the maximum benefit.
The first seven columns are factors, characters and time
stamps that may not have a lot of meaning where correlation and
PCA are concerned. The last column is the outcome.
We will remove these before computing the correlation.



We compute the correlation between the remaining (52) variables,
zero out the diagonal values (those will always be 1) and pick out the ones
that show a correlation of 0.8 or more. As a variable may have a correlation
with more than one variable, we make sure that it occurs only once in our
subset.
Essentially at this point we have split the data set into three parts:

* columns 1:7 & 60 which did not participate in correlation (8 variables)
* columns with correlation less than 0.8 (28 variables)
* columns with correlation 0.8 or more (24 variables)

The last group will be trimmed down using PCA and then recombined with the
other two groups to recreate our dataset. It should be mentioned here that the
same operations are being carried out on the cv-test and final-test sets, too,
so that the final model can be easily applied to those datasets.



We use *preProcess* from the *caret* package with method PCA, and then use the
model with *predict* on all three datasets. *preProcess* automatically selects
the number
of components needed to capture 95% of the variation. In our case this comes
out to 10 variables. That is a reduction of
14 variables.



Now we re-assemble the pieces mentioned earlier and the components selected
from the PCA result. In the first seven columns (that we had removed earlier)
we can see from the names and descriptions that variables
*raw_timestamp_part_1*, *raw_timestamp_part_2*, *cvtd_timestamp*, *new_window*,
*num_window* are time related variables and would be useful if we were doing a
time-series analysis. So in our analysis we choose to drop them. We keep
variable *user_name*.

At this point we still have 40 variables (including the outcome)
which seems to be too
many. The approach we will take now is to fit all these variables into a
*glm* and then use *varImp* to assess their relative importance. We then use
a suitable criteria to decide how many of the variables to use in the final
model.
For the *glm* we will not use *user_name* as it is a factor. *glm* will
split it into multiple variables (equal to the number of levels). That will
have the effect of increasing the number of variables instead of reducing
them. Instead we decide just to include *user_name* in the final list of
variables.



We use *varImp* with the model returned by *glm* and sort the output in
descending order. Using that we re-order our data frame so that higher
column numbers represent decreased importance.

#### Fitting the model

As our outcome *classe* is a factor, a tree-based prediction model seems to
be the right choice. We will try with Recursive Partitioning (rpart) and
Random Forest (randomForest) and see which of these gives a better result.
As Random Forest is quite resource intensive, we will start with a smallish
number of variables. Once the model is finalized, we will try with a larger
number of variables, until we reach a suitably low error rate.



We try both Random Forest and Recursive Partitioning with 18 variables .
We will not be using the *caret* package, instead calling the functions
directly.

The models are applied to the cv-test dataset and the accuracy (and error 
rate) computed. The result is below:





```
##              rpart randomForest
## accuracy 0.6329527   0.98103589
## error    0.3670473   0.01896411
```

We can see that there is considerable difference in the outcomes from
Recursive Partitioning and Random Forest. Random Forest will be our model of
choice for this study.

#### Reducing the Error

We will compute the error rate by using the model to predict
on the cv-test dataset and then comparing the prediction against the
actual classes. The percentage of incorrect predictions will be our
error rate.

With Random Forest and 18 variables we get an estimated Out-of-Sample
error
of 1.9%. We will characterize the Out-of-sample
error estimate
by increasing the number of variables in six steps of two variables, and 
see if we hit a plateau or we reach a reasonable Out-of-sample error rate.

![plot of chunk characerr](figure/characerr-1.png) 

We can see from the plot above that the Estimated Out-of-sample error rate
has reduced monotonically as we increased the number of variables. With
30 variables we have an error rate of
0.918% (or 0.00918 as
a fraction) and an accuracy of 99.082%.

#### Prediction

The final model we select is a Random Forest with the first 30
 most important variables we computed earlier. The list of variables is
given below:


```
##  [1] "user_name"            "magnet_dumbbell_z"    "PC9"                 
##  [4] "accel_forearm_z"      "accel_arm_z"          "PC8"                 
##  [7] "pitch_forearm"        "PC2"                  "PC7"                 
## [10] "magnet_dumbbell_x"    "total_accel_forearm"  "PC4"                 
## [13] "PC5"                  "total_accel_dumbbell" "yaw_arm"             
## [16] "roll_forearm"         "gyros_belt_x"         "PC6"                 
## [19] "pitch_arm"            "gyros_dumbbell_y"     "accel_arm_y"         
## [22] "magnet_forearm_z"     "magnet_forearm_y"     "roll_arm"            
## [25] "magnet_dumbbell_y"    "yaw_forearm"          "magnet_forearm_x"    
## [28] "PC1"                  "accel_forearm_y"      "accel_dumbbell_y"
```

And the predicted outcome on the final-test set is:


```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

Based on the feedback from submission of these results, all the predictions
were correct, giving an effective error rate of 0%.

**Discussion on error rates**

Prediction on the training set gives an accuracy of 100%. This
may be due to overfitting or maybe the sample was very cohesive.

We can get an estimate of the Out-of-Sample error rate from the model itself.
Printing the final model gives:


```
## 
## Call:
##  randomForest(formula = df3$classe ~ ., data = df3[, 1:fvar]) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 0.95%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4182    2    0    0    1 0.0007168459
## B   27 2812    8    1    0 0.0126404494
## C    0   27 2537    3    0 0.0116867939
## D    0    0   48 2357    7 0.0228026534
## E    0    4    3    9 2690 0.0059127864
```

So, the Random Forest function estimates the Out-of-Bag error rate to be 0.95%,
which is pretty low. This error rate is computed by Random Forest using the
0.632 bootstrap algorithm and is supposed to be an unbiased estimate. This is
our expected Out-of-Sample error for the model.

We compared the predicted versus actual classification on the cv-test
dataset, we got an error rate of 0.918%,
which is close to the model's estimated OOB error rate. This is our estimated
Out-of-Sample error from Cross-Validation.

The final error rate as measured from the final-test dataset was 0% for
a sample size of 20. This extremely low error rate may be due to the small
size of the sample.


#### Citations
[1] The data used in this study was collected and made available as a part
of the following publication:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013


---

*-ajtm*

