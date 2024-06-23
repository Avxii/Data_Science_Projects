import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df= pd.read_csv('50_Startups.csv')
print(df.head())
print(df.isnull().sum())
dummy = pd.get_dummies(df["State"])
df=pd.concat([dummy,df],axis=1)
print(df.head())
print(df.columns)
sns.pairplot(df)
plt.show()
x= df[['R&D Spend', 'Administration', 'Marketing Spend']]
y=df['Profit']
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x= ss.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=30)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
mean_absolute_error(y_test,y_pred)
mean_absolute_percentage_error(y_test,y_pred)
print(lr.score(x_test,y_test))
