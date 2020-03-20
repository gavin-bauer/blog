---
title: "Recoding a Logistic Regressor Algorithm"
date: 2020-03-11
lastmod: 2020-03-11
draft: false
description: "I re-implement the main components of a Logistic Regression algorithm before bringing everything together in a Python class."
show_in_homepage: true
show_description: true
license: ''

tags: ['Machine Learning']
categories: ['Machine Learning']

featured_image: ''
featured_image_preview: ''

comment: true
toc: true
autoCollapseToc: true
math: true
---

# 

#### Synopsis
In order to understand the details of the most frequent algorithms in Machine Learning, I have recoded the core concepts in Python using only Numpy (for computations) and Matplotlib (for visualizations).

#### In this post
I re-implement the main components of a Logistic Regression algorithm before bringing everything together in a _Python class_.

#### NB
It is recommended that the reader is familiar with fundamental machine learning concepts such as _features_, _target_, _parameters_, _learning rate_, etc...

## Goal

1. Upload the iris dataset and create a binary classification problem, where \\( y \\) is a discrete value either 0 or 1.

2. Pass the data into a learning algorithm (denoted \\( h \\) for hypothesis).

3. The algorithm outputs a value between 0 and 1, which should be treated as the estimated probability that \\(y = 1 \\) on input x, which can be written as:

     $$h_\theta(x) = P(y=1 | x ; \theta)$$


## Imports


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets # load iris dataset
```

## Create data


```python
iris = datasets.load_iris()
x = iris.data[:, :2] # select only the first 2 features
y = (iris.target != 0) * 1 # make dataset into binary classification problem
```


```python
plt.figure()
plt.scatter(x[:,0], x[:, 1], c=y)
plt.show()
```


![png](/static/images/Logistic-Regression/data.png)


## Hypothesis representation

In the case of a classification problem, the hypothesis (\\( h \\)) can be represented by the function:
\\( h_\theta(x) = g(z) \\), where:
* \\( \theta \\) is a parameter vector.
* \\( x \\) is a feature vector, \\( x_0 \\) always equal to 1.
* \\( g \\) is the sigmoid function written as \\(g(z) = 1 / (1 + e^{-z}) \\)
* with \\(z \\) is equal to \\( \theta^T.x \\)

Combined, these equations give us:

$$h_\theta(x) = 1 / (1 + e^{-\theta^T.x})$$

_Python implementation:_
```
X_b = np.c_[np.ones((100, 1)), x] # add intercept
theta = np.random.rand(X_b.shape[1])
z = X_b.dot(theta)
h = 1 / (1 + np.exp(-z))
```



## Logistic regression components

#### Cost function

A cost function lets us figure out how to fit our data, in this case draw a straight line, by choosing values for \\( \theta \\). Different values of \\( \theta \\) results in different \\( h \\) functions. 

As our goal is to determine which parameters help us bring \\( h \\) closer to \\( y \\), this problem can be formalized as a minimization problem. In the case of a binary classification problem, a commonly used cost function is _"log loss"_, denoted as \\( J(\theta) \\).

$$J(\theta) = -1/m \hspace{1mm} [\sum_{i=1}^{m} \hspace{1mm} y^{(i)} \hspace{1mm} log \hspace{1mm} h_\theta(x^{(i)} ) + (1 - y^{(i)} ) \hspace{1mm} log(1-(h_\theta(x^{(i)} ))]$$

_Python implementation:_
```
m = len(y) # number of instances
J = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
```

#### Gradient Descent

In order to find the best hypothesis, meaning finding the best parameters \\( \hat{\theta} \\) that minize the cost function \\( J(\theta) \\), we will tweak the parameters iteratively using an algorithm called _"Gradient Descent"_ which is written:

$$\theta = \theta - \alpha \hspace{1mm} \partial / \partial \theta J(\theta)$$

The idea is:

1. Start with initial guesses.

2. Keeping changing \\( \theta_0 \\) and \\( \theta_1 \\) by taking big **or** babby steps (aka the _learning rate_ \\( \alpha \\)) to try to reduce \\( J(θ)$.

3. Each time the parameters change, select the gradient (the derative term of \\( J \\)) which reduces \\( J(θ) \\) the most.

4. Repeat.

5. Do so until you converge to a local minimum.

Here, the derivative term used to calculate the gradient is formulated as:

$$\partial J(\theta) / \partial \theta_j =  1/m * X^T(h_{\theta}(x) - y)$$

_Python implementation:_
```
epochs = 1000
learning_rate=0.1
costs = []

for epoch in range(epochs):
    z = X_b.dot(theta)
    h = 1 / (1 + np.exp(-z))
    error = h - y
    gradients = 1/m * X_b.T.dot((error))
    theta = theta - learning_rate * gradients
    
    # optional, checks if J(theta) is minimizing
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    costs.append(cost)
```

## Bringing everything together 

Bringing everything nicely together in a _Python Class_.


```python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=200):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        """
        Compute value of sigmoid function "g", where:
        g(z) = 1 / (1 + e^-z) 
        """
        g = 1 / (1 + np.exp(-z))
        return g
        
    def hypothesis(self, X, theta):
        """
       Compute hypothesis "h" where: h(x) = g(z) with:
            z = X.T * theta
            g, a sigmoid function

		Parameters
    	------------
    	X: numpy array
			Features vector.
    	theta: numpy array 
			Parameters vector.

		Returns
    	------------
		numpy ndarray
		"""
        z = X.dot(theta) 
        h = self.sigmoid(z)
        return h

    def compute_cost(self, h, y):
        """
		Compute value of cost function "J", where:
		J(theta) = -np.average(y * np.log(h) + (1 - y) * np.log(1 - h))

		Parameters
    	------------
    	h: numpy array
			Predictions vector.
		y: numpy array
			Target vector.

		Returns
    	------------
		Value of cost function J
		"""
        m = len(y)
        J = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return J

    def add_intercept(self, X):
        """
        Add intercept to feature vector.
        """
        return np.c_[np.ones((m, 1)), X]


    def fit(self, X, y):
        """
		Fit theta to the training set.

		Parameters
    	------------
    	epochs: int 
			Number of iterations.
    	learning_rate: float 
			Rate at which theta is updated

		Returns
    	------------
		numpy ndarray
		"""
        X_b = self.add_intercept(X)
        costs = []
        self.theta = np.random.rand(X_b.shape[1])
        for epoch in range(self.epochs):
            h = self.hypothesis(X_b, self.theta)
            error = h - y
            gradients = 1/m * X_b.T.dot((error))
            self.theta = self.theta - self.learning_rate * gradients
            cost = self.compute_cost(h, y)
            costs.append(cost)

        self.costs = costs
        self.epochs = epochs

        return self.theta

    def predict(self, X):
        """
        """
        X_b = self.add_intercept(X)
        self.predictions = self.hypothesis(X_b, self.theta)
        return self.predictions

    def plot_cost(self):
        """Plot iterations versus cost function.
        
        Returns
        -----------       
        matploblib figure
        """ 
        plt.figure()
        plt.plot(np.arange(1, self.epochs+1), self.costs)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost function minimization")
        plt.show()

    def plot_model(self, X, y):
        """Plot fitted model.
        
        Returns
        -----------       
        matploblib figure
        """ 
        plt.figure()
        x_values = [np.min(x[:, 0]), np.max(x[:, 1] + 3)]
        y_values = -(self.theta[0] + np.dot(self.theta[1], x_values)) / self.theta[2]
        plt.scatter(x[:,0], x[:,1], c=y)
        plt.plot(x_values, y_values, label='Decision Boundary')
        plt.title("Fitted model")
        plt.show()
  
if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data[:, :2] # select only the first 2 features
    y = (iris.target != 0) * 1 # make dataset into binary classification problem
    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(x, y)

    print("theta_hat", model.theta)

    model.plot_cost()
    model.plot_model(x, y)
```

    theta_hat [-0.61623136  2.94435095 -4.85881711]



![png](/static/images/Logistic-Regression/loss.png)



![png](/static/images/Logistic-Regression/model.png)



```python
# Check with Scikit-Learn's Logistic Regression model
from sklearn import linear_model

log_reg = linear_model.LogisticRegression()
log_reg.fit(x, y.ravel())

print(log_reg.intercept_, log_reg.coef_)

x_values = [np.min(x[:, 0]), np.max(x[:, 1] + 3)]
y_values = -(-8.32330389 + np.dot(3.38829757, x_values)) / -3.1645277
plt.scatter(x[:,0], x[:,1], c=y)
plt.plot(x_values, y_values, label='Decision Boundary')
plt.show()
```

    [-8.32330389] [[ 3.38829757 -3.1645277 ]]



![png](/static/images/Logistic-Regression/scikit-learn.png)

