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
  4. Assign the points for representing the graph.
  5. Predict the regression for marks by using the representation of the graph.
  6. Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Divyashree B S

RegisterNumber:  212221040044

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
print("df.head():")
df.head()

print("df.tail():")
df.tail()

#Segregating data to variables
print("Array value of X:")
X=df.iloc[:,:-1].values
X

print("Array value of X:")
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
print("Values of Y prediction:")
Y_pred

#displaying actual values
print("Array values of Y test:")
Y_test

#graph plot for training data
print("Training set graph:")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
print("Test set graph:")
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("Values of MSE,MAE and RMSE:")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
print('Values of MSE')

```

## Output:

df.head():

\n<img width="219" alt="df head" src="https://user-images.githubusercontent.com/127508123/229509759-3889b858-eec3-4808-abd1-8eafe8adcd84.png">

df.tail():
\n<img width="262" alt="df tail" src="https://user-images.githubusercontent.com/127508123/229510313-85972f36-cc26-42a8-8081-4162ce113514.png">

Array value of X:

<img width="161" alt="x value arr" src="https://user-images.githubusercontent.com/127508123/229510514-675ad971-539b-4048-96aa-47059ced2bb8.png">

Array value of Y:
<img width="551" alt="y value arr" src="https://user-images.githubusercontent.com/127508123/229510767-5a16f387-7468-477c-b099-229a9db9e342.png">

Value of Y prediction:
<img width="612" alt="y pred val" src="https://user-images.githubusercontent.com/127508123/229511116-3d5f0feb-87e3-4177-9731-cbe9932d5734.png">

Array values of Y test:
<img width="418" alt="y test" src="https://user-images.githubusercontent.com/127508123/229511212-f5aa1c8c-c7b9-4d2d-aa17-e26bc27cae2c.png">

Training set Graph:
<img width="394" alt="training set (2)" src="https://user-images.githubusercontent.com/127508123/229511588-e1e83889-812e-44b5-9d20-2175adddd420.png">

Test set Graph:
<img width="362" alt="test set (2)" src="https://user-images.githubusercontent.com/127508123/229511690-63b4b0df-640e-4c93-88ae-c44d702c7119.png">

Values of MSE,MAE,RMSE:
<img width="272" alt="mse,mae,rmse val" src="https://user-images.githubusercontent.com/127508123/229511894-7f4e81ba-9dd9-4bf3-acde-ff6292e2b8f8.png">



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
