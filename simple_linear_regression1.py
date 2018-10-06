#this model will guide you through on how to create simple linear regression model in machine learning using python
#import libraries

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing the dataaset
dataset = pd.read_csv('salary_Data.csv')
x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 1].values


#splitting data set into training and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)

#fitting simple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set results
y_pred = regressor.predict(x_test)

#visualizing training data set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing test data set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
