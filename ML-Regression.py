# Linear Regression

import pandas as pd              #pd is an Alias
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

reg=LinearRegression()

df=pd.read_csv("C:\Users\WIN\Downloads\Data Science 2 Week Complete Data\Datasets\housing.csv")

print(df.isnull().sum())

df['CRIM'].fillna((df['CRIM'].mean()), inplace=True)
df['ZN'].fillna((df['ZN'].mean()), inplace=True)
df['INDUS'].fillna((df['INDUS'].mean()), inplace=True)
df['CHAS'].fillna((df['CHAS'].mean()), inplace=True)
df['AGE'].fillna((df['AGE'].mean()), inplace=True)
df['LSTAT'].fillna((df['LSTAT'].mean()), inplace=True)

x=df.drop('MEDV', axis=1)
y=df['MEDV']

#print(x)
#print(y)

X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.2)

train=reg.fit(X_train,Y_train)

#print(train.coef_)
#print(train.intercept_)

y_pred=reg.predict(X_test)

score=mean_squared_error(Y_test,y_pred)

print(score)