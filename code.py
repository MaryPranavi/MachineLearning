#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[174]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from dataprep.eda import create_report


# In[5]:


#pip install prettytable


# In[3]:


#from prettytable import PrettyTable


# In[23]:


#pip install dataprep


# In[175]:


#from dataprep.eda import create_report


# ## DATA ANALYSIS

# In[176]:


train_df = pd.read_excel('C:/Users/Allam/Desktop/MSCourses/ML/archive/Data_Train.xlsx')


# In[177]:


create_report(train_df)


# ## Exploratory Data Analysis (EDA)

# TRAINING DATA

# In[178]:


train_df.head(10)


# In[179]:


print(train_df.head(10))


# In[180]:


train_df.columns


# In[181]:


train_df.info()


# In[182]:


train_df.describe()


# In[183]:


train_df.isnull().head()


# In[187]:


train_df.isnull().sum()


# In[188]:


train_df.dropna(inplace = True)


# In[189]:


train_df[train_df.duplicated()].head()


# In[190]:


train_df.drop_duplicates(keep='first',inplace=True)
train_df.head()


# In[191]:


train_df.shape


# In[192]:


train_df["Additional_Info"].value_counts()


# In[193]:


train_df["Airline"].unique()


# In[194]:


train_df["Route"].unique()


# TESTING DATA

# In[195]:


#test_df = pd.read_excel("Test_set.xlsx")
test_df = pd.read_excel('C:/Users/Allam/Desktop/MSCourses/ML/archive/Test_set.xlsx')


# In[196]:


test_df.head(10)


# In[197]:


print(test_df.head(10))


# In[198]:


test_df.columns


# In[199]:


test_df.info()


# In[200]:


test_df.describe()


# In[201]:


test_df.isnull().sum()


# ## DATA VISUALIZATION

# Plotting Price vs Airline plot

# In[202]:


sns.catplot(y = "Price", x = "Airline", data = train_df.sort_values("Price", ascending = False), kind="bar", height = 8, aspect = 3)
plt.show()


# Plotting Price vs Source plot

# In[203]:


sns.catplot(y = "Price", x = "Source", data = train_df.sort_values("Price", ascending = False), kind="bar", height = 4, aspect = 3)
plt.show()


# Plotting Price vs Destination plot

# In[204]:


sns.catplot(y = "Price", x = "Destination", data = train_df.sort_values("Price", ascending = False), kind="bar", height = 4, aspect = 3)
plt.show()


# ## DATA EXTRACTION

# In[205]:


train_df.head()


# In[206]:


test_df.head()


# In[207]:


test_df['Duration'] = test_df['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[208]:


train_df['Duration'] = train_df['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[209]:


train_df["Journey_day"] = pd.to_datetime(train_df.Date_of_Journey, format="%d/%m/%Y").dt.day
train_df["Journey_month"] = pd.to_datetime(train_df["Date_of_Journey"], format = "%d/%m/%Y").dt.month
train_df.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[37]:


#train_df.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[210]:


train_df["Dep_hour"] = pd.to_datetime(train_df["Dep_Time"]).dt.hour
train_df["Dep_min"] = pd.to_datetime(train_df["Dep_Time"]).dt.minute
train_df.drop(["Dep_Time"], axis = 1, inplace = True)


# In[211]:


train_df["Arrival_hour"] = pd.to_datetime(train_df["Arrival_Time"]).dt.hour
train_df["Arrival_min"] = pd.to_datetime(train_df["Arrival_Time"]).dt.minute
train_df.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[212]:


train_df.head()


# In[217]:


plt.figure(figsize = (15,4))
plt.title('Price VS Airlines')
plt.xticks(rotation = 90)
sns.scatterplot(y = "Price", x = "Airline", data = train_df.sort_values("Price", ascending = False))
plt.show()


# In[218]:


plt.figure(figsize = (15,4))
plt.title('Price VS Airlines')
plt.scatter(y = "Price", x = "Airline", data = train_df.sort_values("Price", ascending = False))
plt.xlabel('Airline')
plt.ylabel('Price of ticket')
plt.xticks(rotation = 90)
plt.show()


# In[219]:


plt.figure(figsize = (15,15))
sns.heatmap(train_df.corr(), annot = True, cmap = "RdYlGn")
plt.show()


# ## FEATURE ENCODING

# Handling categorical values:

# In[220]:


Airline = train_df[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first= True)


# In[221]:


Source = train_df[["Source"]]
Source = pd.get_dummies(Source, drop_first= True)


# In[222]:


Destination = train_df[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first = True)


# In[223]:


train_df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[224]:


train_df["Total_Stops"].unique()


# In[225]:


train_df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[226]:


print(train_df.columns)


# In[227]:


train_df = pd.concat([train_df, Airline, Source, Destination], axis = 1)


# In[228]:


train_df.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[229]:


train_df.shape[1]


# In[230]:


x=train_df.drop('Price',axis=1)
y=train_df['Price']


# ## MODEL BUILDING

# Data Splitting

# In[231]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# RandomForestRegressor

# In[247]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit (x_train,y_train)
y_pred = model.predict (x_test)

print('Training Score :', model. score(x_train,y_train))
print('Test Score     :', model.score(x_test, y_test))

rf_train_score = model.score(x_train, y_train)
rf_test_score = model.score(x_test, y_test)


# In[248]:


number_of_observations=50
x_ax = range(len(y_test[:number_of_observations]))
plt.plot(x_ax, y_test[:number_of_observations], label="original")
plt.plot(x_ax, y_pred[:number_of_observations], label="predicted")
plt.title("Flight Price test and predicted data")
plt.xlabel('Observation Number')
plt.ylabel('Price')
plt.legend()
plt.show()


# XGBRegressor

# In[249]:


from xgboost import XGBRegressor
model =  XGBRegressor()
model.fit(x_train,y_train)
y_pred =  model.predict(x_test)

print('Training Score :',model.score(x_train, y_train))
print('Test Score     :',model.score(x_test, y_test))

xgb_train_score = model.score(x_train, y_train)
xgb_test_score = model.score(x_test, y_test)


# In[250]:


number_of_observations=50
x_ax = range(len(y_test[:number_of_observations]))
plt.plot(x_ax, y_test[:number_of_observations], label="original")
plt.plot(x_ax, y_pred[:number_of_observations], label="predicted")
plt.title("Flight Price test and predicted data")
plt.xlabel('Observation Number')
plt.ylabel('Price')
plt.legend()
plt.show()


# KNeighborsRegressor

# In[251]:


from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit (x_train,y_train)
y_pred = model.predict (x_test)

print('Training Score :', model. score(x_train,y_train))
print('Test Score     :', model.score(x_test, y_test))

knn_train_score = model.score(x_train, y_train)
knn_test_score = model.score(x_test, y_test)


# In[252]:


number_of_observations=50
x_ax = range(len(y_test[:number_of_observations]))
plt.plot(x_ax, y_test[:number_of_observations], label="original")
plt.plot(x_ax, y_pred[:number_of_observations], label="predicted")
plt.title("Flight Price test and predicted data")
plt.xlabel('Observation Number')
plt.ylabel('Price')
plt.legend()
plt.show()


# DecisionTreeRegressor

# In[253]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit (x_train,y_train)
y_pred = model.predict (x_test)

print('Training Score :', model. score(x_train,y_train))
print('Test Score     :', model.score(x_test, y_test))

dt_train_score = model.score(x_train, y_train)
dt_test_score = model.score(x_test, y_test)


# In[254]:


number_of_observations=50
x_ax = range(len(y_test[:number_of_observations]))
plt.plot(x_ax, y_test[:number_of_observations], label="original")
plt.plot(x_ax, y_pred[:number_of_observations], label="predicted")
plt.title("Flight Price test and predicted data")
plt.xlabel('Observation Number')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[256]:


from tabulate import tabulate


# In[258]:


table = [
    ["Random Forest", rf_train_score, rf_test_score],
    ["XGBoost", xgb_train_score, xgb_test_score],
    ["K-Nearest Neighbors", knn_train_score, knn_test_score],
    ["Decision Tree", dt_train_score, dt_test_score],
]

headers = ["Model", "Training Score", "Test Score"]

# Print the table
print(tabulate(table, headers, tablefmt="fancy_grid"))

