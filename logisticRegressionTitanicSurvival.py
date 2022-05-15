# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:59:08 2022

@author: Vrushabh27
"""

import pandas as pd
import numpy as np
#%%

#Vusualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
#%%

df = pd.read_csv(r'C:\Users\Admin\Downloads\titanic_data.csv')
#%%

df.head()
#%%

# Check for any other unusual values
print(pd.isnull(df).sum())

#%%

# Draw a bar plot of survival by sex
sns.barplot(x = 'Sex', y = 'Survived',data = df)

# print number of females vs. males that survived
print("Number of females who survived: ",df['Survived'][df['Sex'] == 'female'].value_counts())

print("Number of males who survived: ", df['Survived'][df['Sex'] == 'male'].value_counts())
#%%

# Draw a bar plot of survival by sex
sns.barplot(x = 'Sex', y = 'Survived', data = df)

#print percentage females vs. males that survived
print("Percentage females who survived: ",df['Survived'][df['Sex'] == 'female'].value_counts(normalize=True)[1]*100)
print("Percentage males who survived: ",df['Survived'][df['Sex'] == 'male'].value_counts(normalize=True)[1]*100)
#%%
# Draw a bar plot of survival by Pclass
sns.barplot(x = 'Pclass', y = 'Survived', data = df)

#print percentage of survival by Pclass
print("Percentage of Pclass = 1 who survived: ",df['Survived'][df['Pclass'] == 1].value_counts(normalize=True)[1]*100)
print("Percentage of Pclass = 2 who survived: ", df['Survived'][df['Pclass'] == 2].value_counts(normalize=True)[1]*100)
print("Percentage of Pclass = 3 who survived: ", df['Survived'][df['Pclass'] == 3].value_counts(normalize=True)[1]*100)
#%%
# Draw a bar plot SibSp vs. survival
sns.barplot(x = 'SibSp', y = 'Survived', data = df)

#print percentage SibSp survived
print("Percentage of SibSp = 0 who survived: ",df['Survived'][df['SibSp'] == 0].value_counts(normalize=True)[1]*100)
print("Percentage of SibSp = 1 who Survived: ", df['Survived'][df['SibSp'] == 1].value_counts(normalize=True)[1]*100)
print("Percentage of SibSp = 2 who Survived: " ,df['Survived'][df['SibSp'] == 2].value_counts(normalize=True)[1]*100)
print("Percentage of SibSp = 3 who Survived: ", df['Survived'][df['SibSp'] == 3].value_counts(normalize=True)[1]*100)
print("Percentage of SibSp = 4 who Survived: ", df['Survived'][df['SibSp'] == 4].value_counts(normalize=True)[1]*100)
#%%
# Draw a bar plot for Parch vs. Survival
sns.barplot(x = 'Parch', y = 'Survived', data = df)
#%%
plt.figure(figsize=(10,10))
df.Age.value_counts().plot(kind = "pie")
#%%
df['CabinBool'] = (df['Cabin'].notnull().astype('int'))
#calculate percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", df["Survived"][df["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", df["Survived"][df["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x="CabinBool", y="Survived", data=df)
plt.show()
#%%

data = pd.read_csv(r'C:\Users\Admin\Downloads\titanic_data.csv')
data = data.dropna()
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#%%
data = data.drop(['Embarked','Name','Ticket','Cabin'],axis = 1)
#%%
data['Sex'] = data['Sex'].map({'male':0, 'female':1})
#%%
x_train, x_test, y_train, y_test = train_test_split(data.drop(['Survived'],axis = 1),data['Survived'],test_size=0.20,random_state=8)
#%%
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
#%%
accuracy = logmodel.score(x_test, y_test)
print(accuracy*100,'%')
#%%
predictions = logmodel.predict(x_test)
x_test.head()
#%%
print(predictions)
#%%
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
#%%
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))
#%%