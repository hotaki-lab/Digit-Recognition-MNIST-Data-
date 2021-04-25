# Digit Recognition Project (PART-1):

Using the following methods/algorithms for part-1 of this project:

1. Llinear and Logistic Regression, 
2. Non-Linear Features, 
3. Regularization, and 
4. Kernel tricks. 

To see how these methods can be used to solve a real life problem, the famous digit recognition problem using the MNIST (Mixed National Institute of Standards and Technology) database is a perfect project to implement such techniques on.

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project, you will get a chance to experiment with the task of classifying these images into the correct digit using some of the methods indicated above as Part - I of this project.

![alt text](https://github.com/hotaki-lab/Digit-Recognition-Neural-Network/blob/main/Sample%20Digits.PNG "Sample Handwritten Digits")

# Setup Details:

Use Python's **NumPy** numerical library for handling arrays and array operations; use **matplotlib** for producing figures and plots.

Note on software: For all the projects, we will use **python 3.6** augmented with the **NumPy** numerical toolbox, the **matplotlib** plotting toolbox. In this project, we will also use the **scikit-learn** package, which you could install in the same way you installed other packages, e.g. by conda install scikit-learn or pip install sklearn

Download mnist.tar.gz and untar it into a working directory. Use the following Google Drive Link as well:

![alt text](https://drive.google.com/drive/folders/16P4PsmlIqk6FUxFNwJXzLrShrLshf_qw?usp=sharing "Google Drive Containing MNIST folder for this Project")

The archive contains the various data files in the Dataset directory, along with the following python files:

part1/linear_regression.py where you will implement linear regression
part1/svm.py where you will implement support vector machine
part1/softmax.py where you will implement multinomial regression
part1/features.py where you will implement principal component analysis (PCA) dimensionality reduction
part1/kernel.py where you will implement polynomial and Gaussian RBF kernels
part1/main.py where you will use the code you write for this part of the project
Important: The archive also contains files for the second part of the MNIST project. For this project, you will only work with the part1 folder.

To get warmed up to the MNIST data set run python main.py. This file provides code that reads the data from mnist.pkl.gz by calling the function get_MNIST_data that is provided for you in utils.py. The call to get_MNIST_data returns Numpy arrays:

train_x : A matrix of the training data. Each row of train_x contains the features of one image, which are simply the raw pixel values flattened out into a vector of length 784=282. The pixel values are float values between 0 and 1 (0 stands for black, 1 for white, and various shades of gray in-between).

train_y : The labels for each training datapoint, also known as the digit shown in the corresponding image (a number between 0-9).
test_x : A matrix of the test data, formatted like train_x.

test_y : The labels for the test data, which should only be used to evaluate the accuracy of different classifiers in your report.
Next, we call the function plot_images to display the first 20 images of the training set. Look at these images and get a feel for the data (don't include these in your write-up).

Tip: Throughout the whole online grading system, you can assume the NumPy python library is already imported as np. In some problems you will also have access to python's random library, and other functions you've already implemented. Look out for the "Available Functions" Tip before the codebox, as you did in the last project.

This project will unfold both on MITx and on your local machine. However, we encourage you to first implement the functions locally and run the test scripts to validate basic funcitonality. Think of the online graders as a submission box to submit your code when it is ready. You should not have to use the online graders to debug your code.
