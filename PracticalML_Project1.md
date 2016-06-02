# Practical Machine Learning Project
Paul Hickman  

# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell to predict the activity 20 test subjects were performing.

# Libraries

Load the required libraries.


```r
library(caret)
library(rpart)
library(randomForest)
library(gbm)
library(plyr)
library(dplyr)
```

# Data

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.  More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The training data for this project is available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data is available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Loading the Data


```r
# get the training and test data
if (!file.exists("pml-training.csv"))
{
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile="pml-training.csv",method="libcurl")
}
if (!file.exists("pml-testing.csv"))
{
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile="pml-testing.csv",method="libcurl")
}

# load the training and test datasets (removing NA and error values)
training <- read.csv("pml-training.csv", na.strings = c("", "NA", "#DIV/0!"))
testing <- read.csv("pml-testing.csv", na.strings = c("", "NA", "#DIV/0!"))
```

## Tidying the Data

Columns with little or no variability aren't good predictors and will be removed along with the first 7 columns that don't provide any useful information in our analysis.


```r
# remove columns with no data
training <- training[,colSums(is.na(training)) == 0]
testing <- testing[,colSums(is.na(testing)) == 0]

# the first 7 columns aren't required for our analysis
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
```

## Dividing the Data

Now the data has been loaded and in a useable state, the training dataset will be split into a training (60%) and validation (40%) dataset for the purpose of **cross-validation**.


```r
set.seed(9243)
inTrain <- createDataPartition(training$classe, p = 0.6, list = FALSE)
validation <- training[-inTrain, ]
training <- training[inTrain, ]
```

# Building the Models

The models will be built using all available variables in the dataset.  The models to be created are classification tree, random forest and generalised boosted regression.  These models have been selected based on their accuracy with this type of data.  The models will be used against the validation dataset and the model with the highest accuracy will be used with the testing dataset.

## Classification Tree Model


```r
# build model
mRp <- train(classe ~ ., method = "rpart", data = training)

# predict on the validation dataset
pRp <- predict(mRp, validation)

# confusion matrix (for accuracy)
cmRp <- confusionMatrix(pRp, validation$classe)
cmRp
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1990  629  621  572  194
##          B   33  511   49  226  192
##          C  173  378  698  488  373
##          D    0    0    0    0    0
##          E   36    0    0    0  683
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4948          
##                  95% CI : (0.4837, 0.5059)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3405          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8916  0.33663  0.51023   0.0000  0.47365
## Specificity            0.6409  0.92099  0.78203   1.0000  0.99438
## Pos Pred Value         0.4968  0.50544  0.33081      NaN  0.94993
## Neg Pred Value         0.9370  0.85267  0.88319   0.8361  0.89350
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2536  0.06513  0.08896   0.0000  0.08705
## Detection Prevalence   0.5106  0.12886  0.26893   0.0000  0.09164
## Balanced Accuracy      0.7662  0.62881  0.64613   0.5000  0.73401
```

The accuracy of a classification tree model is **49.48%**.

## Random Forest Model

In this model we pass parameters to the training function to perform further **cross-validation** using the training dataset.


```r
# build model
tc <- trainControl(method = "cv", number = 10, allowParallel = FALSE)
mRf <- train(classe ~ ., data = training, method = "rf", trControl = tc)

# predict on the validation dataset
pRf <- predict(mRf, validation)

# confusion matrix (for accuracy)
cmRf <- confusionMatrix(pRf, validation$classe)
cmRf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   12    0    0    0
##          B    3 1503   18    0    0
##          C    0    3 1347   29    4
##          D    0    0    3 1256    5
##          E    0    0    0    1 1433
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9901          
##                  95% CI : (0.9876, 0.9921)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9874          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9901   0.9846   0.9767   0.9938
## Specificity            0.9979   0.9967   0.9944   0.9988   0.9998
## Pos Pred Value         0.9946   0.9862   0.9740   0.9937   0.9993
## Neg Pred Value         0.9995   0.9976   0.9968   0.9954   0.9986
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1916   0.1717   0.1601   0.1826
## Detection Prevalence   0.2856   0.1942   0.1763   0.1611   0.1828
## Balanced Accuracy      0.9983   0.9934   0.9895   0.9877   0.9968
```

The accuracy of a random forest model is **99.01%**.

## Generalised Boosted Regression Model

In this model we pass parameters to the training function to perform further **cross-validation** using the training dataset.


```r
# build model
# v. slow and similar accuracy
tc <- trainControl(method = "cv", number = 10)
mGbm <- train(classe ~ ., data = training, method = "gbm", trControl = tc, verbose = FALSE)

# predict on the validation dataset
pGbm <- predict(mGbm, validation)

# confusion matrix (for accuracy)
cmGbm <- confusionMatrix(pGbm, validation$classe)
cmGbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2197   50    0    1    3
##          B   23 1410   45    8   14
##          C    7   54 1301   37   13
##          D    5    3   20 1230   20
##          E    0    1    2   10 1392
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9597         
##                  95% CI : (0.9551, 0.964)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.949          
##  Mcnemar's Test P-Value : 6.062e-08      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9843   0.9289   0.9510   0.9565   0.9653
## Specificity            0.9904   0.9858   0.9829   0.9927   0.9980
## Pos Pred Value         0.9760   0.9400   0.9214   0.9624   0.9907
## Neg Pred Value         0.9937   0.9830   0.9896   0.9915   0.9922
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2800   0.1797   0.1658   0.1568   0.1774
## Detection Prevalence   0.2869   0.1912   0.1800   0.1629   0.1791
## Balanced Accuracy      0.9874   0.9573   0.9669   0.9746   0.9816
```

The accuracy of a boosted model is **95.97%**.

# Model Selection

The **random forest model** has the highest accuracy and will be selected.  The out of sample error is calculated by 1 - accuracy, which in this case is **0.99%**.

# Testing

The random forest model will be run on the test dataset to predict the category of exercise being performed.


```r
# the predict function is not happy with the data types as they are slightly different
# copying in a row of data from the training data and then removing it was the best
# way to have the testing match the training data types
testing_new <- rbind.fill(training[2, -ncol(training)], testing)
testing_new <- testing_new[-1, ]
```

The answer to the Coursera project is:


```r
# predict on the testing dataset
pTest <- predict(mRf, testing_new)
pTest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
