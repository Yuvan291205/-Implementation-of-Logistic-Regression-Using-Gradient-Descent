# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize Parameters: Set initial values for the weights (w) and bias (b).
2.Compute Predictions: Calculate the predicted probabilities using the logistic function.
3.Compute Gradient: Compute the gradient of the loss function with respect to w and b.
4.Update Parameters: Update the weights and bias using the gradient descent update rule. Repeat steps 2-4 until convergence or a maximum number of iterations is reached. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Yuvan M
RegisterNumber:  212223240188

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee (1).csv")
data.head()

data.info()

data.isnull()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

*/
```

## Output:
![image](https://github.com/Yuvan291205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849170/589f831e-44f9-424b-9353-e87f5bb68c56)
![image](https://github.com/Yuvan291205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849170/b2c92bbd-f6da-4964-98e2-e2ed0d080baa)
![image](https://github.com/Yuvan291205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849170/7f62c49b-6f16-4f7e-b903-b55f096cab42)
![image](https://github.com/Yuvan291205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849170/372b1e4c-7f87-47d6-a26a-2b3a5833bcad)
![image](https://github.com/Yuvan291205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849170/2bf5abc9-898b-4878-a59c-86728996728e)
![image](https://github.com/Yuvan291205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849170/8a92953b-2473-446a-928b-dbe8e5911e4b)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

