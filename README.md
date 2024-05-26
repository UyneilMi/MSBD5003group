# MSBD5003group
This file is the group code for_ HKUST_MSBD_5003_group_. 

## basic structure
Our project builds a complete decision tree and random forest algorithm using basic DataFrame and RDD operations. The code is organized into three Python files: 

_feature\_preprocessing_.py, _impurity\_calculators_.py, and _tree\_based.py_, each handling a specific module. 
Together, these modules enable the training, prediction, and evaluation of decision trees and random forests. The file structure is as follows:

+ feature\_preprocessing.py
   + This module contains preprocessing functions for input data, such as feature scaling, missing value processing, encoding conversion, etc. The preprocessed data will be passed to the model for training and prediction.

+ impurity\_calculators.py
   + Contains various methods for calculating the impurity of decision tree nodes, such as Gini impurity and Entropy. These methods are used to evaluate the quality of different split points to select the optimal split point.
+ tree\_based.py
   + Implementation of decision trees and random forests.


