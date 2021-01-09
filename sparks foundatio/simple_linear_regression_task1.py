#IMPORTING LIBRARIES

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
import urllib as url
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#READING DATA SET
url='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
data=pd.read_csv(url)
#print(data)


#SCATTER PLOT FOR THE ENTIRE DATA SET
data.plot(x='Hours', y='Scores', style='o')
plt.xlabel('HOURS')
plt.ylabel('PERCENTAGE')
plt.show()

#STORING HOURS AND SCORES IN SEPERATE ARRAYS 
X = np.array(data['Hours']).reshape(-1,1)
Y = np.array(data['Scores']).reshape(-1,1)
#print(X,Y)

#SPLITTING TESTING AND TRAINING DATA 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)
#print("training hours data",X_train)
#print("training scores data",Y_train)
#print("testing hours data",X_test)
#print("testing scores data",Y_test)

#FEEDING THE TRAINING DATA TO THE LINEAR REGRESSION ALGORITHM
regR=LinearRegression()
regR.fit(X_train,Y_train)

#print("training completed")


#PLOTTING REGRESSION LINE
line = regR.coef_*X+regR.intercept_
plt.scatter(X, Y)
plt.plot(X, line)
plt.show()


#MAKING PREDICTIONS WITH THE TESTING DATA 
print("testing hours data",X_test)
y_pred = regR.predict(X_test)

print(y_pred, Y_test)


#PREDICTION USING HOURS DATA AS AN INPUT FROM THE USER
print("enter your own data to test ")
test=input()
TEST=np.array(test).reshape(-1,1)
own_pred=regR.predict(TEST)
print("No of Hours =", test)
print("Predicted Score =", own_pred)


