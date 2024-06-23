import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Loading data
test_df= pd.read_csv("test_loan.csv")
train_df= pd.read_csv("train_loan.csv")
print(test_df.head())
print(train_df.head())

##Getting the info about the Dataset
print(train_df.info())
print(test_df.info())

##Getting the statistical info of the Dataset
print(train_df.describe())

##Now we have to predict the loan approval so Loan_ID does not contribute in it
#so we can drop it

# sns.countplot(x=train_df['Gender'])
# plt.show()
# sns.countplot(x=train_df['Married'])
# plt.show()
# sns.countplot(x=train_df['Dependents'])
# plt.show()
# sns.countplot(x=train_df['Education'])
# plt.show()
# sns.countplot(x=train_df['Self_Employed'])
# plt.show()
# sns.countplot(x=train_df['Property_Area'])
# plt.show()
# sns.distplot(x=train_df["ApplicantIncome"])
# plt.show()
# sns.distplot(x=train_df["CoapplicantIncome"])
# plt.show()
# sns.distplot(x=train_df["LoanAmount"])
# plt.show()
# sns.distplot(x=train_df["Loan_Amount_Term"])
# plt.show()
# sns.distplot(x=train_df["Credit_History"])
# plt.show()

print(train_df.isnull().sum())

# filling the missing values for numerical terms - mean
train_df['LoanAmount'] = train_df['LoanAmount'].fillna(train_df['LoanAmount'].mean())
train_df['Loan_Amount_Term'] = train_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].mean())
train_df['Credit_History'] = train_df['Credit_History'].fillna(train_df['Credit_History'].mean())

print(train_df.isnull().sum())

# fill the missing values for categorical terms - mode
train_df['Gender'] = train_df["Gender"].fillna(train_df['Gender'].mode()[0])
train_df['Married'] = train_df["Married"].fillna(train_df['Married'].mode()[0])
train_df['Dependents'] = train_df["Dependents"].fillna(train_df['Dependents'].mode()[0])
train_df['Self_Employed'] = train_df["Self_Employed"].fillna(train_df['Self_Employed'].mode()[0])
print(train_df.isnull().sum())

##data normalisation
#Created a coloumn of total income instead of using two coloumns
train_df['Total_Income'] = train_df['ApplicantIncome']+ train_df["CoapplicantIncome"]

train_df['Total_Income_Log'] = np.log(train_df['Total_Income']+1)
# sns.distplot(train_df["Total_Income_Log"])
# plt.show()
# apply log transformation to the attribute
train_df['ApplicantIncomeLog'] = np.log(train_df['ApplicantIncome']+1)
# sns.distplot(train_df["ApplicantIncomeLog"])
# plt.show()
train_df["CoapplicantIncome_Log"] = np.log(train_df["CoapplicantIncome"]+1)
# sns.distplot(x=train_df["CoapplicantIncome"])
# plt.show()
train_df["LoanAmountLog"] = np.log(train_df["LoanAmount"]+0.5)
# sns.distplot(x=train_df["LoanAmount"])
# plt.show()
train_df["Loan_Amount_Term_Log"] = np.log(train_df["Loan_Amount_Term"])
# sns.distplot(x=train_df["Loan_Amount_Term"])
# plt.show()

#Correlation matrix
data = train_df[["Loan_Amount_Term","LoanAmount","CoapplicantIncome","Total_Income_Log","Total_Income","ApplicantIncome","Credit_History","Loan_Amount_Term_Log","LoanAmountLog","CoapplicantIncome_Log","ApplicantIncomeLog"]]
corr = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap="BuPu")
# plt.show()

#In this graph, the higher density is plotted with dark color and the lower density is plotted with light color.
# We will remove the highly correlated attributes.
# as it is the original attributes and are correlated with log attributes.
# We will remove the previous attributes and keep the log attributes to train our model.
print(train_df.info())
# dropping unnecessary columns
train_df = train_df.drop(columns=['ApplicantIncome', 'CoapplicantIncome','CoapplicantIncome_Log',"Total_Income","LoanAmount", "Loan_Amount_Term",'Loan_ID'], axis=1)
print(train_df.info())
#converting the data
from sklearn.preprocessing import LabelEncoder
cols = ['Gender', "Married", "Education", 'Self_Employed', "Property_Area", "Loan_Status", "Dependents"]
le = LabelEncoder()
for col in cols:
    train_df[col] = le.fit_transform(train_df[col])
print(train_df.info())
print(train_df.head())

#train test split
train = train_df.iloc[:450, :]
test = train_df.iloc[450:, :]
print(train.info())
print(test.info())
#input split
X = train_df.drop(columns=['Loan_Status'], axis=1)
y = train_df['Loan_Status']
print(X.head())

from sklearn.model_selection import train_test_split, cross_val_score


# classify column
def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))

    score = cross_val_score(model, X, y, cv=5)
    print('CV Score:', np.mean(score))

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

#XG boost
from xgboost import XGBClassifier
model = XGBClassifier()
classify(model)

model = LogisticRegression()
model.fit(X, y)
pred = model.predict(X)
print(pred)
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)
print(cm)
