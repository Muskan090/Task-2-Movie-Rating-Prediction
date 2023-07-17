#!/usr/bin/env python
# coding: utf-8

# # Movie Rating Prediction 

# # Importing the Dependencies

# In[8]:


import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# # Data Collection & Processing
# Load movie data

# In[10]:


# Replace 'input_file_path' with the path to your input file
input_file_path = "movies.dat"

# Replace 'output_file_path' with the desired path for the output .csv file
output_file_path = "output_file.csv"

# Read the .dat file into a DataFrame using a specific encoding
df_movie = pd.read_csv(input_file_path, encoding='latin1',sep = '::', engine='python',header=None)

# Write the DataFrame to a .csv file
df_movie.to_csv(output_file_path, index=False)
df_movie.head(20)


# In[11]:


df_movie.columns =['MovieIDs','MovieName','Genre']
df_movie.dropna(inplace=True)
df_movie.head()


# In[12]:


df_movie.shape


# In[13]:


df_movie.info()


# # Handling the Missing values

# In[14]:


df_movie.isnull().sum()


# **Load the rating data** 

# In[15]:


# Replace 'input_file_path' with the path to your input file
input_file_path = "ratings.dat"

# Replace 'output_file_path' with the desired path for the output .csv file
output_file_path = "output_file.csv"

# Read the .dat file into a DataFrame using a specific encoding
df_rating = pd.read_csv(input_file_path, encoding='latin1',sep = '::', engine='python',header=None)

# Write the DataFrame to a .csv file
df_rating.to_csv(output_file_path, index=False)
df_rating.head(20)


# In[16]:


df_rating.columns =['ID','MovieID','Ratings','TimeStamp']
df_rating.dropna(inplace=True)
df_rating.head()


# In[17]:


df_rating.shape


# In[18]:


df_rating.info()


# **Handling the Missing values**

# In[19]:


df_rating.isnull().sum()


# **Load User data**

# In[20]:


# Replace 'input_file_path' with the path to your input file
input_file_path = "users.dat"

# Replace 'output_file_path' with the desired path for the output .csv file
output_file_path = "output_file.csv"

# Read the .dat file into a DataFrame using a specific encoding
df_user = pd.read_csv(input_file_path, encoding='latin1',sep = '::', engine='python',header=None)

# Write the DataFrame to a .csv file
df_user.to_csv(output_file_path, index=False)
df_user.head(20)


# In[21]:


df_user.columns =['UserID','Gender','Age','Occupation','Zip-code']
df_user.dropna(inplace=True)
df_user.head()


# In[22]:


df_user.shape


# In[23]:


df_user.info()


# **Handling the Missing values**

# In[24]:


df_user.isnull().sum()


# ## Combine the all dataset 

# In[25]:


# merge all 3 data set 
df = pd.concat([df_movie, df_rating,df_user], axis=1)
df.head()


# # Data Analysis
# 

# In[26]:


#Visualize overall rating by users
df['Ratings'].value_counts().plot(kind='bar',alpha=0.7,figsize=(10,10))
plt.show()


# In[27]:


df.Age.plot.hist(bins=15)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('Age')


# In[28]:


movies = df.groupby('MovieName').size().sort_values(ascending=True)[:1000]
print(movies)


# # Perform LogisticRegression on first 700 extracted records

# In[29]:


first_700 = df[500:]
first_700.dropna(inplace=True)


# In[30]:


#Use the following features:movie id,age,occupation
features = first_700[['MovieID','Age','Occupation']].values


# In[31]:


#Use rating as label
labels = first_700[['Ratings']].values


# In[32]:


# Create train and test data set
train, test, train_labels, test_labels = train_test_split(features,labels,test_size=0.33,random_state=42)


# In[33]:


#Create a histogram for movie
df.Age.plot.hist(bins=15)
plt.title("Movie & Rating")
plt.ylabel('MovieID')
plt.xlabel('Ratings')


# In[34]:


#Create a histogram for age
df.Age.plot.hist(bins=15)
plt.title("Age & Rating")
plt.ylabel('Age')
plt.xlabel('Ratings')


# In[35]:


logreg = LogisticRegression()
logreg.fit(train, train_labels)
Y_pred = logreg.predict(test)
log = round(logreg.score(train, train_labels) * 100, 2)
log

