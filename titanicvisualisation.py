
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

##lets plot the data before processing it
#plotting analysing the categorical values
sns.countplot(x=df['Survived'])
plt.show()
sns.countplot(x=df['Pclass'])
plt.show()
sns.countplot(x=df['Sex'])
plt.show()
sns.countplot(x=df['SibSp'])
plt.show()
sns.countplot(x=df['Parch'])
plt.show()
sns.countplot(x=df['Embarked'])
plt.show()
#Now analysing the numerical values

sns.distplot(x=df['Age'])
plt.show()
sns.distplot(x=df['Fare'])
plt.show()

##Now we have to do the normalisation of the data
##Lets check for the missing values
df.isnull().sum()







