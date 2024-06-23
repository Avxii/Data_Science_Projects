
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

  # 'TkAgg' is a common choice, but you can also try 'Qt5Agg', 'Agg', etc.

import warnings
warnings.filterwarnings('ignore')

#loading data

df= pd.read_csv("tested.csv")
print(df.head())


##Getting the info about the Dataset
print(df.info())

##Getting the statistical info of the Dataset
print(df.describe())

##From here we can see the datatypes of the coulumns we have to make changes are Embarked
#Cabin, Age, Sex and Name (Name cannot be changed into any usefule integer value so we can drop it)
#Same with the cabin as it is not much of use we can drop it

##Now we have to do the normalisation of the data
##Lets check for the missing values
print(df.isnull().sum())

#we will drop the cabin coloumn as it has more than 60 perecnt null values
df = df.drop(columns=['Cabin'], axis=1)
# we will replace the null values in age with the mean
#we will do the same for fare
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
print(df.isnull().sum())

##log data disribution for normalisation of Age
df['Age'] = np.log(df['Age']+1)
sns.distplot(df['Age'])
# plt.show()
##log data disribution for normalisation of Fare
df['Fare'] = np.log(df['Fare']+2)
sns.distplot(df['Fare'])
# plt.show()

## drop unnecessary columns
df = df.drop(columns=['Name', 'Ticket','PassengerId'], axis=1)
print(df.head())

## now converting the datatypes for the machine to read
from sklearn.preprocessing import LabelEncoder
cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
print(df.head())
# here we converted the male and female to 0 and 1
#and embarked to S,C,Q to 0,1,2


#Train-Test Split
train = df.iloc[:350, :]
test = df.iloc[350:, :]
print(train.tail())
print(test.head())
# input split
X = train.drop(columns=['Survived'], axis=1)
y = train['Survived']
print(X.head())


#training the module
from sklearn.model_selection import train_test_split


# classify column
def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))

#Linear Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model)


#Prediction
model = RandomForestClassifier()
model.fit(X, y)

pred = model.predict(X)
print(pred)






