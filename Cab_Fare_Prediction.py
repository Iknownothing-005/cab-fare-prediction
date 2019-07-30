#!/usr/bin/env python
# coding: utf-8

# # Cab Fare Prediction 
# 
# Importing Data and Libraries 

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime 

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


# In[2]:


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[3]:


data = pd.read_csv("train_cab.csv")
data_test = pd.read_csv("test.csv")


# In[4]:


data.head()


# In[5]:


data.info()


# # Exploratory Data Analysis
# 
# Finding Outliers and Data distribution
# 

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

import pylab as plot
params = {
    "axes.labelsize":"large",
    "xtick.labelsize":"x-large",
    "legend.fontsize": 20,
    "figure.dpi": 150,
    "figure.figsize": [25,7]
}
plot.rcParams.update(params)

labelsize = 15
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize


# In[7]:


# converting fare_amount coulmn to float 
data["fare_amount"] = data["fare_amount"].apply(pd.to_numeric,errors='coerce')


# In[8]:


data_plot1 = [data["pickup_longitude"].dropna(),data["dropoff_longitude"].dropna()]
data_plot2 = [data["pickup_latitude"].dropna(),data["dropoff_latitude"].dropna()]
data_plot3 = [data["passenger_count"].dropna()]
data_plot4 = [data["fare_amount"].dropna()]

x1 = ["pickup_longitude","dropoff_longitude"]
x2 = ["pickup_latitude","dropoff_latitude"]
x3 = ["passenger_count"]
x4 = ["fare_amount"]


fig = plt.figure(1, figsize = (25,30))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)



plot = ax1.boxplot(data_plot1, patch_artist = True, vert = False)
plot = ax2.boxplot(data_plot2, patch_artist = True, vert = False)
plot = ax3.boxplot(data_plot3, patch_artist = True, vert = False)
plot = ax4.boxplot(data_plot4, patch_artist = True, vert = False) 

ax1.set_yticklabels(x1)
ax2.set_yticklabels(x2)
ax3.set_yticklabels(x3)
ax4.set_yticklabels(x4)

plt.savefig('BoxPlot.png')
plt.show()


# In[9]:


# since Signed Degree Format latitude ranges from -90 to 90 there by removing outlier 
data = data[~(data["pickup_latitude"] > 90)]
data = data[~(data["pickup_latitude"] < -90)]
data = data[~(data["dropoff_latitude"] > 90)]
data = data[~(data["dropoff_latitude"] < -90)]


# In[10]:


# since Signed Degree Format longitude ranges from -180 to 180 thererby removing outliers
data = data[~(data["pickup_longitude"] > 180)]
data = data[~(data["pickup_longitude"] < -180)]
data = data[~(data["dropoff_longitude"] > 180)]
data = data[~(data["dropoff_longitude"] < -180)]


# In[11]:


#Since maximum passenger in a cab can be 7 therefore removing all the values greater than that 
data = data[~(data["passenger_count"] > 7)]


# In[12]:


# dropping the fare amount above 500 dollars
data = data[~(data["fare_amount"] > 400)]


# In[13]:


p = 0.017453292519943295     #Pi/180
data["distance"] = 0.5 - np.cos((data["dropoff_latitude"] - data["pickup_latitude"]) * p)/2 + np.cos(data["pickup_latitude"]* p) * np.cos(data["dropoff_latitude"]* p) * (1 - np.cos((data["dropoff_longitude"] -data["pickup_longitude"]) * p)) /2
data["distance"] = 12742 * np.arcsin(np.sqrt(data["distance"])) #2*R*asin...


# In[14]:


p = 0.017453292519943295     #Pi/180
data_test["distance"] = 0.5 - np.cos((data_test["dropoff_latitude"] - data_test["pickup_latitude"]) * p)/2 + np.cos(data_test["pickup_latitude"]* p) * np.cos(data_test["dropoff_latitude"]* p) * (1 - np.cos((data_test["dropoff_longitude"] -data_test["pickup_longitude"]) * p)) /2
data_test["distance"] = 12742 * np.arcsin(np.sqrt(data_test["distance"])) #2*R*asin...


# In[15]:


data.head()


# In[16]:


data_plot1 = [data["pickup_longitude"].dropna(),data["dropoff_longitude"].dropna()]
data_plot2 = [data["pickup_latitude"].dropna(),data["dropoff_latitude"].dropna()]
data_plot3 = [data["passenger_count"].dropna()]
data_plot4 = [data["fare_amount"].dropna()]



x1 = ["pickup_longitude","dropoff_longitude"]
x2 = ["pickup_latitude","dropoff_latitude"]
x3 = ["passenger_count"]
x4 = ["fare_amount"]



fig = plt.figure(1, figsize = (25,30))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)



plot = ax1.boxplot(data_plot1, patch_artist = True, vert = False)
plot = ax2.boxplot(data_plot2, patch_artist = True, vert = False)
plot = ax3.boxplot(data_plot3, patch_artist = True, vert = False)
plot = ax4.boxplot(data_plot4, patch_artist = True, vert = False) 


ax1.set_yticklabels(x1)
ax2.set_yticklabels(x2)
ax3.set_yticklabels(x3)
ax4.set_yticklabels(x4)

plt.savefig('BoxPlot.png')
plt.show()


# In[17]:


#plotting density distibution of continous variables

fig, axes = plt.subplots(2,1, figsize = (25,10))

sns.distplot(list(data["passenger_count"].dropna()), ax = axes[0], color = "y")
sns.distplot(list(data["fare_amount"].dropna()), ax = axes[1], color = "B")


axes[0].set(xlabel = "PASSENGER_COUNT")
axes[1].set(xlabel = "FARE_AMOUNT")


plt.savefig('dist1.png')
plt.show()


# # DATA CLEANING AND FEATURE ENGINEERING

# In[18]:


#Importing KNN library for missing value imputation 
from fancyimpute import KNN


# In[19]:


data.info()


# In[20]:


#counting number of zeros for each column 
zero_counts = (data == 0).sum(axis=0)
zero_counts


# In[21]:


col = list(data.columns)
for i in col:
    data[col] = data[col].replace(0,np.nan)


# In[22]:


#counting number of zeros for each column 
zero_counts = (data == 0).sum(axis=0)
zero_counts


# In[23]:


#checking missing value in train data
missing_values = pd.DataFrame(data.isnull().sum())
missing_values


# In[24]:


#checking missing value in test data
missing_values1 = pd.DataFrame(data_test.isnull().sum())
missing_values1


# In[25]:


#stripping the word UTC from column pickup_datetime 
data["pickup_datetime"] = data["pickup_datetime"].str.replace("UTC","")

#converting column pickup_datetime to datetime format  
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors = "coerce")

#extract date and time from pickup_datetime column and creating separate column for each
data["pickup_weekday"] = data["pickup_datetime"].dt.weekday_name
data["pickup_hour"] = data["pickup_datetime"].dt.hour
data["pickup_DayOfMonth"] = data["pickup_datetime"].dt.day
data["pickup_month"] = data["pickup_datetime"].dt.month_name().str.slice(stop=3)
data["pickup_year"] = data["pickup_datetime"].dt.year

data.drop(["pickup_datetime"], axis = 1, inplace = True)

data.head()


# In[26]:


#feature engineering on test data 
#stripping the word UTC from column pickup_datetime 
data_test["pickup_datetime"] = data_test["pickup_datetime"].str.replace("UTC","")

#converting column pickup_datetime to datetime format  
data_test["pickup_datetime"] = pd.to_datetime(data_test["pickup_datetime"], errors = "coerce")

#extract date and time from pickup_datetime column and creating separate column for each
data_test["pickup_weekday"] = data_test["pickup_datetime"].dt.weekday_name
data_test["pickup_hour"] = data_test["pickup_datetime"].dt.hour
data_test["pickup_DayOfMonth"] = data_test["pickup_datetime"].dt.day
data_test["pickup_month"] = data_test["pickup_datetime"].dt.month_name().str.slice(stop=3)
data_test["pickup_year"] = data_test["pickup_datetime"].dt.year

data_test.drop(["pickup_datetime"], axis = 1, inplace = True)

data_test.head()


# In[27]:


#coverting segregrating hours to group and creating dummies for data 
data["Morning"] = data["pickup_hour"].map(lambda x: 1 if 12 > x >=6 else 0 )
data["afternoon"] = data["pickup_hour"].map(lambda x: 1 if 12 <= x < 17 else 0 )
data["evening"] = data["pickup_hour"].map(lambda x: 1 if 17 <= x < 20 else 0 )
data["night"] = data["pickup_hour"].map(lambda x: 1 if 20 <= x < 23 else 0 )
data["late_night"] = data["pickup_hour"].map(lambda x: 1 if 0 <= x < 3 else 0 )
data["early_morning"] = data["pickup_hour"].map(lambda x: 1 if 3 <= x < 6 else 0 )


# In[28]:


#coverting segregrating hours to group and creating dummies for data_test
data_test["Morning"] = data_test["pickup_hour"].map(lambda x: 1 if 12 > x >=6 else 0 )
data_test["afternoon"] = data_test["pickup_hour"].map(lambda x: 1 if 12 <= x < 17 else 0 )
data_test["evening"] = data_test["pickup_hour"].map(lambda x: 1 if 17 <= x < 20 else 0 )
data_test["night"] = data_test["pickup_hour"].map(lambda x: 1 if 20 <= x < 23 else 0 )
data_test["late_night"] = data_test["pickup_hour"].map(lambda x: 1 if 0 <= x < 3 else 0 )
data_test["early_morning"] = data_test["pickup_hour"].map(lambda x: 1 if 3 <= x < 6 else 0 )


# In[29]:


data.shape


# In[30]:


#creating dummies for all categorical classes 
def get_dummies(dataset,columns):
    for j in columns:
        dummies = pd.get_dummies(dataset[j], prefix = j)
        dataset = pd.concat([dataset, dummies], axis = 1)
        dataset.drop(j,axis =1, inplace = True)
    return dataset 


# In[31]:


cat_col = ["pickup_weekday","pickup_month","pickup_year"]
data = get_dummies(data,cat_col)


# In[32]:


data.rename(columns={'pickup_year_2009.0':'pickup_year_2009',
                          'pickup_year_2010.0':'pickup_year_2010',
                          'pickup_year_2011.0':'pickup_year_2011',
                          'pickup_year_2012.0':'pickup_year_2012',
                          'pickup_year_2013.0':'pickup_year_2013',
                          'pickup_year_2014.0':'pickup_year_2014',
                          'pickup_year_2015.0':'pickup_year_2015'}, 
                 inplace=True)


# In[33]:


data.head()


# In[34]:


#creating dummies for test data categorical variables 
data_test = get_dummies(data_test,cat_col)
data_test.head()


# In[35]:


#dropping one dummy variable from each categorical feature which has least counts and thereby avoiding dummy variable
data.drop(["pickup_weekday_Monday"],axis = 1, inplace = True)
data.drop(["pickup_month_Aug"],axis = 1,inplace = True)
data.drop(["early_morning"],axis = 1,inplace = True)
data.drop(["pickup_year_2015"],axis = 1,inplace = True)
data.drop(["pickup_hour"],axis = 1,inplace = True)


# In[36]:


#test_data
#dropping one dummy variable from each categorical feature which has least counts and thereby avoiding dummy variable 
data_test.drop(["pickup_weekday_Monday"],axis = 1, inplace = True)
data_test.drop(["pickup_month_Aug"],axis = 1,inplace = True)
data_test.drop(["early_morning"],axis = 1,inplace = True)
data_test.drop(["pickup_year_2015"],axis = 1,inplace = True)
data_test.drop(["pickup_hour"],axis = 1,inplace = True)


# In[90]:


data.iloc[0:5,8:36]


# In[38]:


data_test.shape


# # MISSING VALUE IMPUTATION 

# In[39]:


missing_values = pd.DataFrame(data.isnull().sum())
missing_values


# In[40]:


data.info()


# In[41]:


data = pd.DataFrame(KNN(k=7).fit_transform(data), columns = data.columns)
data.info()


# In[42]:


data.iloc[:,[5,7]] = data.iloc[:,[5,7]].round()


# In[43]:


data.shape


# In[44]:


missing_values = pd.DataFrame(data.isnull().sum())
missing_values


# In[45]:


#normalizing values in data 
col = data.columns
for i in col.drop("fare_amount"):
    print(i)
    data[i] = (data[i] - min(data[i]))/ (max(data[i])- min(data[i]))


# In[46]:


#normalizing values in data_test
col = data_test.columns
for i in col:
    print(i)
    data_test[i] = (data_test[i] - min(data_test[i]))/ (max(data_test[i])- min(data_test[i]))


# In[47]:


data_test.shape


# # CHECKING LINEAR RELATION BETWEEN DEPENDEDNT AND INDIPENDENT VARIABLES

# In[48]:


fig, axes = plt.subplots(2,2, figsize = (25,10))

g = sns.regplot(x="passenger_count", y="fare_amount",
               data=data, ax = axes[0][0], color = "y",label="passenger_count")
g = sns.regplot(x="distance", y="fare_amount",
               data=data, ax = axes[0][1], color ="b", label = "distance")
g = sns.regplot(x="pickup_longitude", y="fare_amount",
               data=data, ax = axes[1][0], color = "r", label = "pickup_longitude")
g = sns.regplot(x="dropoff_longitude", y="fare_amount",
               data=data, ax = axes[1][1], color = "m", label = "pickup_DayOfMonth" )


axes[0][0].set(xlabel = "passenger_count")
axes[0][1].set(xlabel = "distance")
axes[1][0].set(xlabel = "pickup_longitude")
axes[1][1].set(xlabel = "pickup_DayOfMonth")

axes[0][0].legend(loc="best")
axes[0][1].legend(loc="best")
axes[1][0].legend(loc="best")
axes[1][1].legend(loc="best")

plt.savefig("reg1.png")
plt.show()


# # FEATURE SELECTION

# In[49]:


from scipy.stats import chi2_contingency


# In[50]:


correlation_data = data.iloc[:,1:8]
correlation_data.head()


# In[51]:


# plotting heat map
f, ax = plt.subplots(figsize=(7,5))

#creating correlation matrix
corr_matrix = correlation_data.corr()

#plotting 
sns.heatmap(corr_matrix, mask = np.zeros_like(corr_matrix, dtype = np.bool), 
            cmap = sns.diverging_palette(220,10,as_cmap=True), square = True, ax=ax,linewidths=.5, annot=True)
plt.show()


# In[52]:


categorical_col = list(data.iloc[:,8:36])
print(categorical_col)
for i in categorical_col:
    print (i)
    chi2,p,dof,ex = chi2_contingency(pd.crosstab(data["fare_amount"],data[i]))
    print(p)


# In[53]:


data_reduced = data.iloc[:,[0,5,6,7,9,12,13,22,25,26,28,29,30,31,32,33,34,35]]


# In[54]:


data_reduced.head()


# In[55]:


print(data_test.columns)
print(data_reduced.columns)


# In[56]:


data_test_reduced = data_test.iloc[:,[4,5,6,8,11,12,21,24,25,27,28,29,30,31,32,33,34]]


# In[57]:


print (data_reduced.shape)


# In[58]:


data_test_reduced.head()
print (data_test_reduced.shape)


# # TRAIN AND TESTING MODEL

# IMPORTING LIBRARIES AND MODEL 

# In[59]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import statsmodels.formula.api as sm 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


# In[60]:


X = data.iloc[:,1:36]
Y = data.iloc[:,0:1]


# In[61]:


# Usng Backward Elimination method with multiple linear 
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_opt = X[cols]
    X_opt = np.append(arr = np.ones((16042,1)).astype(int), values = X_opt,axis=1)
    Reg_OLS = sm.OLS(Y,X_opt).fit()
    p = pd.Series(Reg_OLS.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

selected_features_BE = cols
print(selected_features_BE)


# In[62]:


X_final = data.loc[:,selected_features_BE]
X_final.shape


# In[63]:


#splitting X_opt into training and test data 
x_train,x_test,y_train,y_test = train_test_split(X_final,Y,test_size=0.35,random_state = 0)


# In[64]:


Reg_OLS_final = sm.OLS(endog = y_train, exog = x_train).fit()
y_pred =Reg_OLS_final.predict(x_test)


# In[65]:


rsme = np.sqrt(mean_squared_error(y_test,y_pred))
print (rsme)


# In[66]:


r2_score(y_test,y_pred)


# In[67]:


X1 = data_reduced.iloc[:,1:18].values
Y1 = data_reduced.iloc[:,0:1].values


# In[68]:


#splitting X_opt into training and test data 
x_train1,x_test1,y_train1,y_test1 = train_test_split(X1,Y1,test_size=0.35,random_state = 0)


# In[69]:


svr_reg = SVR()
svr_reg.fit(x_train1,np.ravel(y_train1))


# In[70]:


y_pred1 =svr_reg.predict(x_test1)


# In[71]:


rsme = np.sqrt(mean_squared_error(y_test1,y_pred1))
print (rsme)


# In[72]:


r2_score(y_test1,y_pred1)


# In[73]:


RF_reg = RandomForestRegressor(random_state=0)
RF_reg.fit(x_train1,np.ravel(y_train1))


# In[74]:


y_pred2 =RF_reg.predict(x_test1)


# In[75]:


rsme = np.sqrt(mean_squared_error(y_test1,y_pred2))
print (rsme)


# In[76]:


r2_score(y_test1,y_pred2)


# In[77]:


#calling classification models
log_reg = LogisticRegression()
RF_classifier = RandomForestClassifier()


# In[78]:


y_train1 = np.ravel(y_train1).astype("int")
models = [RF_classifier,log_reg]
for model in models:
    print (model)
    print("accuracy",":",cross_val_score(estimator = model,X= x_train1 , y = y_train1,scoring = "accuracy",cv=10,n_jobs=-1).mean())


# In[79]:


#using grid search to optimize Random Forest Regression model
parameters = [{"n_estimators":[70,75,80],"max_depth":[12,15,20],"min_samples_leaf":[10,12,15]}]


# In[80]:


grid_search = GridSearchCV(estimator = RF_reg,
                          param_grid = parameters,
                          scoring =  "r2",
                          cv = 10,
                          n_jobs = -1)


# In[81]:


grid_search = grid_search.fit(x_train1,np.ravel(y_train1))


# In[82]:


best_accuracy = grid_search.best_score_


# In[83]:


best_accuracy


# In[84]:


best_parameters = grid_search.best_params_


# In[85]:


best_parameters 


# In[86]:


RF_reg_otpimized = RandomForestRegressor(max_depth = 20, n_estimators = 70, min_samples_leaf = 10, random_state=0)
RF_reg_otpimized.fit(x_train1,np.ravel(y_train1))


# In[87]:


y_pred4 = RF_reg_otpimized.predict(x_test1)


# In[88]:


r2_score(y_test1,y_pred4) 


# In[89]:


rsme = np.sqrt(mean_squared_error(y_test1,y_pred4))
print (rsme)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script config_template.ipynb')

