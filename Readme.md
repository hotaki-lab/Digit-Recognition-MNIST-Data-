# Digit Recognition Project (PART-1):

Using the following methods/algorithms for part-1 of this project:

1. Linear and Logistic Regression, 
2. Non-Linear Features, 
3. Regularization, and 
4. Kernel tricks. 

To see how these methods can be used to solve a real life problem, the famous digit recognition problem using the MNIST (Mixed National Institute of Standards and Technology) database is a perfect project to implement such techniques on.

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project, you will get a chance to experiment with the task of classifying these images into the correct digit using some of the methods indicated above as Part - I of this project.

![alt text](https://github.com/hotaki-lab/Digit-Recognition-Neural-Network/blob/main/Sample_Digits.PNG "Sample Handwritten Digits")

# Setup Details:

Use Python's **NumPy** numerical library for handling arrays and array operations; use **matplotlib** for producing figures and plots.

Note on software: For all the projects, we will use **python 3.6** augmented with the **NumPy** numerical toolbox, the **matplotlib** plotting toolbox. In this project, we will also use the **scikit-learn** package, which you could install in the same way you installed other packages, e.g. by conda install scikit-learn or pip install sklearn

Download mnist.tar.gz and untar it into a working directory. Use the following Google Drive Link as well:

[Google Drive Containing MNIST folder for this Project](https://drive.google.com/drive/folders/16P4PsmlIqk6FUxFNwJXzLrShrLshf_qw?usp=sharing)

The archive contains the various data files in the Dataset directory, along with the following python files:

* part1/linear_regression.py 
  * where you will implement linear regression
* part1/svm.py 
  * where you will implement support vector machine
* part1/softmax.py 
  * where you will implement multinomial regression
* part1/features.py 
  * where you will implement principal component analysis (PCA) dimensionality reduction
* part1/kernel.py 
  * where you will implement polynomial and Gaussian RBF kernels
* part1/main.py 
  * where you will use the code you write for this part of the project

Important: The archive also contains files for the second part of the MNIST project. For this project, you will only work with the part1 folder.

To get warmed up to the MNIST data set run python main.py. This file provides code that reads the data from mnist.pkl.gz by calling the function get_MNIST_data that is provided for you in utils.py. The call to get_MNIST_data returns Numpy arrays:

* train_x : A matrix of the training data. Each row of train_x contains the features of one image, which are simply the raw pixel values flattened out into a vector of length 784=282. The pixel values are float values between 0 and 1 (0 stands for black, 1 for white, and various shades of gray in-between).
* train_y : The labels for each training datapoint, also known as the digit shown in the corresponding image (a number between 0-9).
* test_x : A matrix of the test data, formatted like train_x.
* test_y : The labels for the test data, which should only be used to evaluate the accuracy of different classifiers in your report.

Next, we call the function plot_images to display the first 20 images of the training set. Look at these images and get a feel for the data (don't include these in your write-up).

![alt text](https://github.com/hotaki-lab/Digit-Recognition-Neural-Network/blob/main/Figure_1.png "Displaying First 20 Images of Training Set")

[Introduction to ML Packages (Part-1)](https://github.com/Varal7/ml-tutorial/blob/master/Part1.ipynb)

# Linear Regression with Closed Form Solution:

It can be argued that we can apply a linear regression model, as the labels are numbers from 0-9. Though being a little doubtful, you decide to have a try and start simple by using the raw pixel values of each image as features.

There is a skeleton code **run_linear_regression_on_MNIST in main.py**, but it needs more to complete the code and make the model work. The following will cover how to complete this code:

To solve the linear regression problem, you recall the linear regression has a closed form solution:

![alt text](https://github.com/hotaki-lab/Digit-Recognition-Neural-Network/blob/main/Linear_Regression.JPG "Linear Regression")

```python 
def closed_form(X, Y, lambda_factor):
"""
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
"""
    xt = X.transpose()
    xtx = xt @ X
    id = np.eye(xtx.shape[0])
    lam_id = lambda_factor * id
    inner = (xtx + lam_id)
    inner_inv = np.linalg.inv(inner)
    outer = xt @ Y
    theta = (inner_inv) @ (outer)
    return theta
    raise NotImplementedError
```

# Test Error on Linear Regression:

Apply the linear regression model on the test set. For classification purpose, you decide to round the predicted label into numbers 0-9.

Note: For this project we will be looking at the error rate defined as the fraction of labels that don't match the target labels, also known as the "gold labels" or ground truth. (In other context, you might want to consider other performance measures such as precision and recall).

The test error of the linear regression algorithm for different λ (the output from the main.py run).

* **Error (λ = 1) = 0.7697**
* **Error (λ = 0.1) = 0.7698**
* **Error (λ = 0.01) = 0.7702**

# What went Wrong?

* **We found that no matter what λ factor we try, the test error is LARGE.**
* **The loss function related to the closed-form solution is inadequate for this problem.**
* **The closed form solution of linear regression is the solution of optimizing the mean squared error loss.**
* **This is not an appropriate loss function for a classification problem.**

# Applying Support Vector Machine (SVM) Algorithm:

It is found above that it is clearly not a regression problem, but a classification problem. We can change it into a binary classification and use the SVM to solve the problem. In order to do so, it is suggested that we build a **one vs. rest model** for every digit. For example, classifying the digits into two classes: 0 and not 0.

Function **run_svm_one_vs_rest_on_MNIST** considers changing the labels of digits 1-9 to 1 and keeps the label 0 for digit 0. The **scikit-learn package** contains an SVM model that can be used directly.

We will be working in the file **part1/svm.py** in this problem

Important: For this problem, the **scikit-learn library** to be used. If you don't have it, install it using **pip install sklearn**

# One vs. Rest SVM:

Use the **sklearn** package and build the SVM model on your local machine. Use **random_state = 0, C=0.1** and default values for other parameters.

```python
import sklearn

def one_vs_rest_svm(train_x, train_y, test_x):   
"""
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
"""
    clf = sklearn.svm.LinearSVC(C=0.1 ,random_state=0)
    clf.fit(train_x, train_y)
    pred_test_y = clf.predict(test_x)
    return pred_test_y
    
    raise NotImplementedError
```

# Binary Classification Error:

Report the test error by running **run_svm_one_vs_rest_on_MNIST**.

**Error = 0.007499999999999951**

# Applying C-SVM:

Play with the C parameter of SVM. The following statements are true about the C parameter:

* **Larger C gives smaller tolerance of violation.**
* **Larger C gives a smaller-margin separating hyperplane.**

# Applying Multiclass SVM:

In fact, **sklearn** already implements a **multiclass SVM** with a one-vs-rest strategy. Use **LinearSVC** to build a multiclass SVM model:

```python
import sklearn

def multi_class_svm(train_x, train_y, test_x):   
"""
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
"""
    clf = sklearn.svm.LinearSVC(C=0.1 ,random_state=0)
    clf.fit(train_x, train_y)
    pred_test_y = clf.predict(test_x)
    return pred_test_y
    
    raise NotImplementedError
```

# Multiclass SVM Error:

Report the overall test error by running **run_multiclass_svm_on_MNIST**

**Error = 0.08189999999999997**

# Multinomial (Softmax) Regression & Gradient Descent:

Instead of building ten models, we can expand a single logistic regression model into a multinomial regression and solve it with similar gradient descent algorithm.

The main function which you will call to run the code you will implement in this section is **run_softmax_on_MNIST** in **main.py** (already implemented). In the following steps, a number of the methods are described that are already implemented in **softmax.py** that will be useful.

In order for the regression to work, you will need to implement three methods. Below we describe what the functions should do. Some test cases are included in **test.py** to help verify that the methods implemented are behaving sensibly.

We will be working in the file **part1/softmax.py** in this problem:

# Computing Probabilities for Softmax:

Writing a function **compute_probabilities** that computes, for each data point x(i), the probability that x(i) is labeled as j for j=0,1,…,k−1.

The softmax function **h** for a particular vector x requires computing:

![alt text](https://github.com/hotaki-lab/Digit-Recognition-Neural-Network/blob/main/Softmax.JPG "Linear Regression")

```python
def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """

    itemp = 1 / temp_parameter
    dot_products = itemp * theta.dot(X.T)
    max_of_columns = dot_products.max(axis=0)
    shifted_dot_products = dot_products - max_of_columns
    exponentiated = np.exp(shifted_dot_products)
    col_sums = exponentiated.sum(axis=0)
    return exponentiated / col_sums
    
    raise NotImplementedError
```

# Cost Function:

Write a function **compute_cost_function** that computes the total cost over every data point.

The cost function J(θ) is given by: (Use natural log)

![alt text](https://github.com/hotaki-lab/Digit-Recognition-Neural-Network/blob/main/cost.JPG "Cost Function")

```
def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """

    N = X.shape[0]
    probabilities = compute_probabilities(X, theta, temp_parameter)
    selected_probabilities = np.choose(Y, probabilities)
    non_regulizing_cost = np.sum(np.log(selected_probabilities))
    non_regulizing_cost *= -1 / N
    regulizing_cost = np.sum(np.square(theta))
    regulizing_cost *= lambda_factor / 2.0
    return non_regulizing_cost + regulizing_cost
    
    raise NotImplementedError
```

# Gradient Descent:

The function **run_gradient_descent_iteration** is necessary for the rest of the project.

Now, in order to run the gradient descent algorithm to minimize the cost function, we need to take the derivative of J(θ) with respect to a particular θm. Notice that within J(θ), we have:

![alt text](https://github.com/hotaki-lab/Digit-Recognition-Neural-Network/blob/main/GD.JPG "GD")

![alt text](https://github.com/hotaki-lab/Digit-Recognition-Neural-Network/blob/main/GD1.JPG "GD")
