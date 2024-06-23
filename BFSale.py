import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("train_bfsale.csv")
test= pd.read_csv("test_bfsale.csv")
print(train.head())

##Getting the info about the Dataset

print(train.info())
print(train.describe())

## lets plot the attributes

# sns.countplot(x=train['Gender'])
# plt.show()
# sns.countplot(x=train['Age'])
# plt.show()
# sns.countplot(x=train['City_Category'])
# plt.show()
# sns.countplot(x=train['Stay_In_Current_City_Years'])
# plt.show()
#
# sns.distplot(x=train['Occupation'])
# plt.show()
# sns.distplot(x=train['Marital_Status'])
# plt.show()
# sns.distplot(x=train['Product_Category_1'])
# plt.show()
# sns.distplot(x=train['Product_Category_2'])
# plt.show()
# sns.distplot(x=train['Product_Category_3'])
# plt.show()
# sns.distplot(x=train['Purchase'])
# plt.show()

##finding missing values
print(train.isnull().sum())

# Replace using median
median = train['Product_Category_2'].median()
train['Product_Category_2'].fillna(median, inplace=True)
print(train.isnull().sum())
# dropping Product_Category_3 beacause it has a larger number of null values
train=train.drop('Product_Category_3',axis=1)
print(train.isnull().sum())
train.hist(edgecolor='black',figsize=(12,12));
plt.show()

##drop the useless attributes
train = train.drop(['Product_ID','User_ID'],axis=1)

df_Gender = pd.get_dummies(train['Gender'])
df_Age = pd.get_dummies(train['Age'])
df_City_Category = pd.get_dummies(train['City_Category'])
df_Stay_In_Current_City_Years = pd.get_dummies(train['Stay_In_Current_City_Years'])

data_final= pd.concat([train, df_Gender, df_Age, df_City_Category, df_Stay_In_Current_City_Years], axis=1)

print(data_final.head())

data_final = data_final.drop(['Gender','Age','City_Category','Stay_In_Current_City_Years'],axis=1)
print(data_final.info())

from sklearn.model_selection import train_test_split


x=data_final.drop('Purchase',axis=1)
y=data_final.Purchase
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=30)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train, y_train)
print(lm.fit(x_train, y_train))
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)
print('Intercept parameter:', lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])
print(coeff_df)
predictions = lm.predict(x_test)
print("Predicted purchases (in dollars) for new costumers:", predictions)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))