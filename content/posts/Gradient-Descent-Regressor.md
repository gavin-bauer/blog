---
title: "Recoding Gradient Descent Regressor"
date: 2020-03-05
lastmod: 2020-03-05
draft: false
description: "I re-implement the main components of a Gradient Descent Regressor algorithm before bringing everything together in a Python class."
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

#### Synopsis
In order to understand the details of the most frequent algorithms in Machine Learning, I have recoded the core concepts in Python using only Numpy (for computations) and Matplotlib (for visualizations).

#### In this post
I re-implement the main components of a Gradient Descent Regressor algorithm before bringing everything together in a _Python class_.

#### NB
It is recommended that the reader is familiar with fundamental machine learning concepts such as _features_, _target_, _parameters_, _learning rate_, etc...


## Goal

1. Create some data using a linear function \\( y \\), where \\( y = 3 + 5x + noise \\).

2. Pass the data into a learning algorithm.

3. Algorithm outputs a function (denoted \\( h \\) for hypothesis) where the equation is as close to \\( y \\) as possible, where \\( h(x) = \theta_0 + \theta_1x \\).


## Imports


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
```

## Notations

\\( h(x) \\) can be written more concisely in a vectorized form: \\( h(x) = \theta . x \\), where:
* \\( \theta \\) is a parameter vector, containing the bias (3) and the coefficient (5).
* \\( x \\) is a feature vector, \\( x_0 \\) always equal to 1.
* "." is the dot product of the vectors \\( \theta \\) and \\( x \\). To make it possible to multiply these vectors together, a column of 1s will be added to \\( x \\).

_Python implementation:_
```
X_b = np.c_[np.ones((100, 1)), x]
h = X_b.dot(theta)
```


## Create data


```python
x = 2 * np.random.rand(100, 1)
y = 3 + 5 * x + np.random.rand(100, 1)

X_b = np.c_[np.ones((100, 1)), x] 
theta = np.random.rand(2, 1)
```

## Implement regressor components

#### Cost function

A cost function lets us figure out how to fit our data, in this case draw a straight line, by choosing values for \\( \theta \\). Different values of \\( \theta \\) results in different \\( h \\) functions. 

As our goal is to determine which parameters help us bring \\( h \\) closer to \\( y \\), this problem can be formalized as a minimization problem. In the case of a regression, a commonly used cost function is the _"Mean Squared Error"_, denoted as \\( J(\theta) \\).

$$J(\theta) = 1/m * \sum_{i=1}^{m}(h(x)^i_\theta - y^i)^2$$

_Python implementation:_
```
m = len(y)
error = h - y
J = (1/m) * np.sum(error**2)
```

#### Gradient Descent

In order to find the best hypothesis, meaning finding the best parameters \\( \hat{\theta} \\), we will tweak the parameters iteratively using an algorithm called _"Gradient Descent"_ which is written:

$$\theta = \theta - \alpha \partial / \partial \theta J(\theta)$$

The idea is:

1. Start with initial guesses.

2. Keeping changing \\( \theta_0 \\) and \\( \theta_1 \\) by taking big **or** babby steps (aka the _learning rate_ \\( \alpha \\)) to try to reduce \\( J(θ) \\).

3. Each time the parameters change, select the gradient (the derative term of \\( J \\)) which reduces \\( J(θ) \\) the most.

4. Repeat.

5. Do so until you converge to a local minimum.

Here, the cost function is the Mean Squared Error and its derivative term used to calculate the gradient is formulated as:

$$\partial / \partial \theta J(\theta) =  2/m * X^T(h(x) - y)$$

_Python implementation:_
```
gradients = 2/m * X_b.T.dot(error) 
theta = theta - learning_rate * gradients
```

## Training the model


```python
epochs=100
learning_rate=0.1

m = len(y)
theta = np.random.rand(2, 1)
X_b = np.c_[np.ones((m, 1)), x]
costs = []

for epoch in range(epochs):
    h = X_b.dot(theta)
    error = h - y
    gradients = 2/m * X_b.T.dot(error) # slope of the derivative of the cost function
    theta = theta - learning_rate * gradients # update theta after each iteration
    cost = (1/m) * np.sum(error**2)
    costs.append(cost)

costs = costs
epochs = epochs
predictions = X_b.dot(theta)

print(theta)
```

    [[3.4765757 ]
     [5.02021439]]
    

After training, the optimal parameters found \\( \hat{\theta} \\) are: `3.47`and `5.02`.

These parameters are very close to the actual function used to generate the data: \\( y = 3 + 5x + noise \\).


```python
# Check with Scikit-Learn's Linear Regression model
from sklearn import linear_model

lin_reg = linear_model.LinearRegression()
lin_reg.fit(x, y.ravel())

print(lin_reg.intercept_, lin_reg.coef_)
```

    3.462283486675374 [5.03315005]
    

## Bringing everything together 

Bringing everything nicely together in a _Python Class_.


```python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class GradientDescentRegressor(object):
	"""
	GradientDescent

    Parameters
    ------------
	X: numpy array
		Features vector.
	y: numpy array
		Target vector.

	Attributes
	------------
	theta : 1-d numpy array, shape = [polynomial order + 1,] 
		Parameters randomly initialized, with theta[0] corresponding
		to the intercept term
	
	method : str , values = "batch_gradient_descent" | "SGD" | "MBGD"
		Method used for finding optimal values of theta
	
	If gradient descent method is chosen:
	
		costs : 1-d numpy array,
			Cost function values for every iteration of gradient descent
		
		epochs: int
			Number of iterations of gradient descent to be performed
    """

	def __init__(self, X, y):
		self.X = X
		self.y = y
	
	def hypothesis(self, X, theta):
		"""
		Compute hypothesis "h" where:
		h(x) = theta_0 * x + theta_1 

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
		h = X.dot(theta)
		return h
	
	def compute_cost(self, X, y, theta):
		"""
		Compute value of cost function "J", where:
		J(theta) = 1/m * ((X.dot(theta)) - y)**2

		Parameters
    	------------
    	X: numpy array
			Features vector.
		y: numpy array
			Target vector.
		theta: numpy array
			Parameters vector.

		Returns
    	------------
		Value of cost function J
		"""
		m = len(y)
		h = self.hypothesis(X, theta)
		error = h - y
		J = (1/m) * np.sum(error**2)
		return J
    
	def fit(self, epochs=100, learning_rate=0.01):
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
		m = X.shape[0]
		theta = np.random.rand(2, 1)
		X_b = np.c_[np.ones((m, 1)), X]
		costs = []
		
		for epoch in range(epochs):
			h = self.hypothesis(X_b, theta)
			error = h - y
			gradients = 2/m * X_b.T.dot(error) # slope of the derivative of the cost function
			theta = theta - learning_rate * gradients # update parameters after each iteration
			cost = self.compute_cost(X_b, y, theta)
			costs.append(cost)

		self.costs = costs
		self.epochs = epochs
		self.predictions = X_b.dot(theta)

		return theta

	def plot_cost(self):
		"""Plot number of gradient descent iterations versus cost function.
        
        Returns
        -----------       
        matploblib figure
        """ 
		plt.figure()
		plt.plot(np.arange(1, self.epochs+1), self.costs)
		plt.show()

	def plot_model(self):
		"""Plot number of gradient descent iterations versus cost function.
        
        Returns
        -----------       
        matploblib figure
        """ 
		plt.figure()
		plt.scatter(X, y)
		plt.plot(X, self.predictions, "g-")
		plt.show()

if __name__ == "__main__":
	X = 2 * np.random.rand(100, 1)
	y = 3 + 5 * X + np.random.rand(100, 1)

	gradient_descent = GradientDescentRegressor(X, y)

	print(gradient_descent.fit())
	gradient_descent.plot_cost()
	gradient_descent.plot_model()
```

    [[3.47976772]
     [4.90424675]]
    
