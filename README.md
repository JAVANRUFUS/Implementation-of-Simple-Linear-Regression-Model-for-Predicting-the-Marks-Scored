# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
# javan rufus j
# 212224230104

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Data Preparation: Collect a dataset of study hours (g) and corresponding marks (k), then split them into training and testing sets.

2.Model Training: Calculate the slope (o) and intercept (p) using the Ordinary Least Squares method to minimize the sum of squared differences between predicted and actual values.

3.Prediction: Apply the learned parameters to new input data using the linear equation: " K = oG + p".

4.Evaluation: Assess the model's performance by calculating metrics such as Mean Squared Error (MSE) or R^2 score on the test set.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: javan rufus j
RegisterNumber:212224230104  
*/






import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([35, 50, 65, 70, 85])
model = LinearRegression()
model.fit(X, Y)
m = model.coef_[0]
b = model.intercept_
print("Slope (m):", m)
print("Intercept (b):", b)
x = float(input("Enter the number of hours studied: "))
marks = model.predict([[x]])
print("Predicted Marks for the Number of hours studied is:", marks[0])
Y_pred = model.predict(X)
plt.scatter(X, Y, label=" Actual Data points")
plt.plot(X, Y_pred, label="Regression Line")
plt.xlabel("No. of Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Implementation of Simple Linear Regression Model for Predicting the Marks Scored")
plt.legend()
plt.show()


```

## Output:

<img width="1247" height="650" alt="image" src="https://github.com/user-attachments/assets/89083426-5c67-453c-8c24-2455fcb8519d" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
