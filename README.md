# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for Gradient Design. 

2. Set variables for assigning dataset values. 

3. Import linear regression from sklearn.

4. Assign the points for representing the graph.                                                                                                                5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Naadira Sahar N

RegisterNumber: 212221220034
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

#Segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

#displaying actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
print('Values of MSE')
```

## Output:
1.df.head()

![df head()](https://user-images.githubusercontent.com/128135126/229293862-34870b9a-d46d-41cb-9ea0-13cea22f00ae.png)

2.df.tail()

![df tail()](https://user-images.githubusercontent.com/128135126/229293967-09c262a2-1793-4fa3-9ad4-5b969c669900.png)

3.Array value of X

![Array value of X](https://user-images.githubusercontent.com/128135126/229293469-8f08eb90-7774-46aa-b0db-0ac8df78dad3.png)

4.Array value of Y

![Array value of Y](https://user-images.githubusercontent.com/128135126/229293494-aa427d62-0d42-4747-9b9d-474c5f58fb29.png)

5.Values of Y prediction

![Values of Y prediction](https://user-images.githubusercontent.com/128135126/229293514-ef09d849-1b86-4783-b366-9552fbafecca.png)

6.Array values of Y test

![Array values of Y test](https://user-images.githubusercontent.com/128135126/229293545-32b41b3c-8494-4138-8f49-6161ed6af60b.png)

7.Training set graph

![Training set graph](https://user-images.githubusercontent.com/128135126/229293565-fbd372b3-aac6-4ed6-be0b-87905f046ebb.png)

8.Test set graph

![Test set graph](https://user-images.githubusercontent.com/128135126/229293588-9a4d9a34-3f38-4e5f-bd77-3e3f79bb2cf0.png)

9.Values of MSE,MAE and RMSE

![Values of MSE,MAE and RMSE](https://user-images.githubusercontent.com/128135126/229293605-f9e791d8-b7c0-45c1-ac1b-2901364e8b26.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
