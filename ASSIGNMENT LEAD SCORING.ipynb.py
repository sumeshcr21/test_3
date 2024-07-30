#!/usr/bin/env python
# coding: utf-8

# ## Lead Scoring Assignment

# In[1]:

 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


Leaddata = pd.read_csv("D:\my files\MS in Data Analytics\Assignment Lead Scoring\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv")
Leaddata.head()


# In[3]:


Leaddata.describe()


# In[4]:


Leaddata.info()


# In[5]:


round(Leaddata.isnull().mean()*100,2)


# * Large number of columns has null values and hence should be dropped

# In[6]:


Leaddata.shape


# In[7]:


Leaddata.duplicated().sum()


# # EDA

# ### DATA  CLEANING

# #### Removing unnecessary columns

# In[8]:


# Dropping Prsopect ID and Lead Number as they are just identity of individuals and won't be helpful in anakysis

Leaddata.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[9]:


# Dropping Columns which has no relevance for our model

Leaddata.drop(['Tags','Lead Quality','Last Activity','Last Notable Activity'],axis=1,inplace=True)


# #### Replacing select with null values, as it means the user didn't select any  options at all from drop down menu

# In[10]:


# Select all non-numeric columns
Leaddata_obj = Leaddata.select_dtypes(include='object')

# Find out columns that have "Select"
s = lambda x: x.str.contains('Select', na=False)
l = Leaddata_obj.columns[Leaddata_obj.apply(s).any()].tolist()
print (l)


# In[11]:


# select all the columns that have a "Select" entry
select_cols = ['Specialization', 'How did you hear about X Education', 'Lead Profile', 'City']

# replace values
Leaddata[select_cols] = Leaddata[select_cols].replace('Select', np.NaN)


# #### Dealing with null values

# In[12]:


#Checking percentage of null values

round((Leaddata.isnull().mean()*100),2)


# In[13]:


#Removing null values greater than 40% 

Leaddata = Leaddata.loc[:, Leaddata.isnull().mean()*100 < 40]
Leaddata.columns


# In[14]:


#rechecking remaining null values

round((Leaddata.isnull().mean()*100),2)


# Removing the rest null values would mean losing lot of important data and hence we will invidividually check those columns

# ### Now dealing will columns having null values greater than 10%

# In[15]:


#Listing remaining null values greater than 10%

Leaddata.loc[:, Leaddata.isnull().mean()*100 > 10].columns


# #### Selecting and dealing the above columns one by one

# ##### COUNTRY COLUMN

# In[16]:


#Checking percentage value of items in 'Country' column

round(Leaddata.Country.value_counts(normalize = True, dropna = False) * 100,2)


# In[17]:


#If we replace null value with mode, then the column would be highly asymmetric and hence we can remove this column

Leaddata.drop('Country', axis = 1, inplace = True)


# ##### SPECIALIZATION COLUMN

# In[18]:


#Checking percentage value of items in 'Specialization' column

round(Leaddata.Specialization.value_counts(normalize = True, dropna = False) * 100,2)


# In[19]:


courses=Leaddata.Specialization.value_counts().index.tolist()
courses


# In[20]:


#Categorising and merging fields into management, business and industry courses

management_courses = []
business_courses = []
industry_courses = []


for course in courses:
    if 'management' in course.lower():
        management_courses.append(course)
    elif 'business' in course.lower():
        business_courses.append(course)
    else:
        industry_courses.append(course)
        


# In[21]:


print(management_courses)
print(business_courses)
print(industry_courses)


# In[22]:


Leaddata.loc[Leaddata['Specialization'].isin(management_courses), 'Specialization'] = 'Management Specializations'
Leaddata.loc[Leaddata['Specialization'].isin(business_courses), 'Specialization'] = 'Business Specializations'
Leaddata.loc[Leaddata['Specialization'].isin(industry_courses), 'Specialization'] = 'Industry Specializations'


# In[23]:


# replacing null value with not provided as missing value is significantly high
Leaddata['Specialization'] = Leaddata.Specialization.fillna('Not Provided')


# In[24]:


round(Leaddata['Specialization'].value_counts(normalize=True,dropna=True)*100,2)


# ##### WHAT IS YOUR CURRENT OCCUPATION COLUMN

# In[25]:


#Checking percentage value of items in 'What is your current occupation' column

round(Leaddata['What is your current occupation'].value_counts(normalize = True, dropna = False) * 100,2)


# In[26]:


# Combining columns with negligible percenatge to one column
Leaddata.loc[(Leaddata['What is your current occupation'] == 'Student') | (Leaddata['What is your current occupation'] == 'Other') | (Leaddata['What is your current occupation'] == 'Housewife') | (Leaddata['What is your current occupation'] == 'Businessman') , 'What is your current occupation'] = 'Student and Others'


# In[27]:


round(Leaddata['What is your current occupation'].value_counts(normalize = True) * 100,2)


# In[28]:


# Imputting null values with Proportionate value obtained from other column 

Leaddata['What is your current occupation'] = Leaddata['What is your current occupation'].fillna(pd.Series(np.random.choice(['Unemployed', 'Working Professional','Student and Others'],p = [0.8550, 0.1078, 0.0372], size = len(Leaddata))))


# In[29]:


round(Leaddata['What is your current occupation'].value_counts(normalize=True)*100,2)


# ##### WHAT MATTERS MOST TO YOU IN CHOOSING A COURSE COLUMN

# In[30]:


Leaddata['What matters most to you in choosing a course'].value_counts(normalize=True)*100


# In[31]:


#This column also appears highly asymmetric and hence we can remove the same

Leaddata.drop('What matters most to you in choosing a course', axis=1, inplace=True)


# ##### CITY COLUMN

# In[32]:


round(Leaddata.City.value_counts(normalize=True,dropna=False)*100,2)


# In[33]:


#As most of the people didn't select city, we will fill null values with 'Not Provided'

Leaddata['City'] = Leaddata['City'].fillna('Not provided')


# In[34]:


#We will group the above columns into Mumbai, Other Maharashtra cities, and Other states city

#First combinig Maharastrian Cities
Leaddata.City.loc[(Leaddata.City=='Thane & Outskirts') | (Leaddata.City=='Other Cities of Maharashtra')]='Other Maharashtrian Cities'

#Now combining remaining cities
Leaddata.City.loc[(Leaddata.City=='Other Cities') | (Leaddata.City=='Other Metro Cities') | (Leaddata.City=='Tier II Cities')]='Non-Maharashtrian Cities'


# In[35]:


round(Leaddata.City.value_counts(normalize=True)*100,2)


# #### DEALING WITH REMAINING COLUMNS

# ##### LEAD SOURCE COLUMN

# In[36]:


#Checking percentage of items in this column

round(Leaddata['Lead Source'].value_counts(normalize=True,dropna=False)*100,2)


# In[37]:


#Inputting the missing value with mode as it is categorical variable

Leaddata['Lead Source'].fillna(Leaddata['Lead Source'].mode()[0],inplace=True)


# In[38]:


#Also combining many smallers values into new columns named 'Websites'

Leaddata['Lead Source'] = Leaddata['Lead Source'].apply(lambda x: x if ((x== 'Google') | (x=='Direct Traffic') | (x=='Olark Chat') | (x=='Organic Search') | (x=='Reference')) else 'Websites')


# In[39]:


#Rechecking percetage value in Lead source column

round(Leaddata['Lead Source'].value_counts(normalize=True,dropna=False)*100,2)


# ##### TOTALVISITS COLUMN

# In[40]:


#Checking percentage value of items in column

round(Leaddata['TotalVisits'].value_counts(normalize=True,dropna=False)*100,2)


# In[41]:


#As it is numerical column replacing the missing value with median

Leaddata['TotalVisits'].fillna(Leaddata['TotalVisits'].median(), inplace=True)
Leaddata['TotalVisits'] = Leaddata['TotalVisits'].astype('int') #converted to int because number of visits can't be decimal


# ##### PAGE VIEWS PER VISIT COLUMN

# In[42]:


#Checking percentage value of items in column

round(Leaddata['Page Views Per Visit'].value_counts(normalize=True,dropna=False)*100,2)


# In[43]:


#Similar to 'TotalVisits' column replacing missing value with median in this column too

Leaddata['Page Views Per Visit'].fillna(Leaddata['Page Views Per Visit'].median(), inplace=True)
Leaddata['Page Views Per Visit'] = Leaddata['Page Views Per Visit'].astype('int') #converted to int because 'page views per visit' can't be decimal


# In[44]:


#Making sure that no null value exists

round(Leaddata.isnull().mean()*100,2)


# #### SORTING OUT BINARY COLUMNS

# In[45]:


# determine unique values
for a, b in Leaddata.select_dtypes(include='object').nunique().to_dict().items():
    print('{} = {}'.format(a,b))


# In[46]:


#dropping columns which have just one unique entry as it can't help our model

Leaddata.drop(['Magazine','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque'],axis=1,inplace=True)


# In[47]:


# Putting rest of binary columns in another dataframe
Leaddata_bin = Leaddata[['Do Not Email', 'Do Not Call', 'Search', 'Newspaper Article', 'X Education Forums','Newspaper', 'Digital Advertisement', 'Through Recommendations', 'A free copy of Mastering The Interview']]

# Checking Imbalance in data with the help of value_count percentage
for i in Leaddata_bin.columns:
    x = round((Leaddata_bin[i].value_counts(normalize = True)) * 100,2)
    print(x)
    print('_____________________________________________________________')
   


# In[48]:


# we can drop all columns except 'Do Not Email' and 'A free copy of Mastering The Interview'

drop_columns=['Do Not Call', 'Search', 'Newspaper Article', 'X Education Forums','Newspaper', 'Digital Advertisement', 'Through Recommendations']

Leaddata.drop(drop_columns, axis = 1, inplace = True)


# In[49]:


Leaddata.info()


# ### EXPLORATORY DATA ANALYSIS

# #### CATEGORICAL COLUMNS

# In[50]:


plt.figure(figsize = (12, 8))

Leaddata['Lead Origin'].value_counts().sort_values(ascending = False).plot(kind= 'barh', width = 0.8,edgecolor = 'black')

plt.show()


# In[51]:


plt.figure(figsize = (12, 8))

Leaddata['Lead Source'].value_counts().sort_values(ascending = False).plot(kind= 'barh', width = 0.8,edgecolor = 'black')

plt.show()


# In[52]:


plt.figure(figsize = (12, 8))

Leaddata['Do Not Email'].value_counts().sort_values(ascending = False).plot(kind= 'barh', width = 0.8,edgecolor = 'black')

plt.show()


# In[53]:


plt.figure(figsize = (12, 8))

Leaddata['Specialization'].value_counts().sort_values(ascending = False).plot(kind= 'barh', width = 0.8,edgecolor = 'black')

plt.show()


# * Most people were interested in management specialization
# 

# In[54]:


plt.figure(figsize = (12, 8))

Leaddata['What is your current occupation'].value_counts().sort_values(ascending = False).plot(kind= 'barh', width = 0.8,edgecolor = 'black')

plt.show()


# * Most people who applied were unemployed

# In[55]:


plt.figure(figsize = (12, 8))

Leaddata['City'].value_counts().sort_values(ascending = False).plot(kind= 'barh', width = 0.8,edgecolor = 'black')

plt.show()


# In[56]:


plt.figure(figsize = (12, 8))

Leaddata['A free copy of Mastering The Interview'].value_counts().sort_values(ascending = False).plot(kind= 'barh', width = 0.8,edgecolor = 'black')

plt.show()


# #### NUMERICAL COLUMNS

# In[57]:


#Checking distribution 

fig = plt.figure(figsize = (14, 10))
plt.subplot(2, 2, 1)
plt.hist(Leaddata['TotalVisits'], bins = 20)
plt.title('Total website visits')

plt.subplot(2, 2, 2)
plt.hist(Leaddata['Total Time Spent on Website'], bins = 20)
plt.title('Total Time Spent on Website')

plt.subplot(2, 2, 3)
plt.hist(Leaddata['Page Views Per Visit'], bins = 20)
plt.title('Number of page views per visit')

plt.show()


# * Asymmetricity indicating presence of outliers

# In[58]:


#Checking outliers

plt.figure(figsize = (12, 8))

plt.subplot(3,1,1)
sns.boxplot(Leaddata['TotalVisits'])

plt.subplot(3,1,2)
sns.boxplot(Leaddata['Total Time Spent on Website'])

plt.subplot(3,1,3)
sns.boxplot(Leaddata['Page Views Per Visit'])
plt.show()


# * It can be clearly seen that there are outliers in 'TotalVisits' and 'Page views per visit' columns

# In[59]:


plt.figure(figsize = (12,10))
sns.heatmap(Leaddata[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']].corr(), annot = True)
plt.show()


# Collinearity is not very high, hence we can move ahead without removing column

# #### NOW COMPARING CATEGORICAL VARIABLES WITH CONVERTED COLUMN

# In[60]:


plt.figure(figsize = (20,20))

plt.subplot(3,3,1)
sns.countplot(x='Lead Origin', hue='Converted', data= Leaddata).tick_params(axis='x', rotation = 90)
plt.title('Lead Origin')

plt.subplot(3,3,2)
sns.countplot(x='Lead Source', hue='Converted', data= Leaddata).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[61]:


plt.figure(figsize = (20,20))
plt.subplot(2,2,2)
sns.countplot(x='Specialization', hue='Converted', data= Leaddata).tick_params(axis='x', rotation = 90)
plt.title('Specialization')
plt.show()

plt.figure(figsize = (20,20))
plt.subplot(2,2,4)
sns.countplot(x='What is your current occupation', hue='Converted', data= Leaddata).tick_params(axis='x', rotation = 90)
plt.title('What is your current occupation')
plt.show()

plt.figure(figsize = (20,20))
plt.subplot(2,2,4)
sns.countplot(x='City', hue='Converted', data= Leaddata).tick_params(axis='x', rotation = 90)
plt.title('City')
plt.show()


# In[62]:


plt.figure(figsize = (20,20))

plt.subplot(3,3,1)
sns.countplot(x='Do Not Email', hue='Converted', data= Leaddata).tick_params(axis='x', rotation = 90)
plt.title('Do Not Email ')


# In[63]:


plt.figure(figsize = (20,20))

plt.subplot(3,3,1)
sns.countplot(x='A free copy of Mastering The Interview', hue='Converted', data= Leaddata).tick_params(axis='x', rotation = 90)
plt.title('A free copy of Mastering The Interview')


# ### OUTLIERS

# In[64]:


numeric = Leaddata[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]
numeric.describe(percentiles=[0.25,0.5,0.75,0.9,0.99])


# In[65]:


plt.figure(figsize = (8,8))
sns.boxplot(y=Leaddata['TotalVisits'])
plt.show()


# In[66]:


plt.figure(figsize = (8,8))
sns.boxplot(y=Leaddata['Total Time Spent on Website'])
plt.show()


# In[67]:


plt.figure(figsize = (8,8))
sns.boxplot(y=Leaddata['Page Views Per Visit'])
plt.show()


# In[68]:


# capping at 99 percentile
Leaddata['TotalVisits'].loc[Leaddata['TotalVisits'] >= Leaddata['TotalVisits'].quantile(0.99)] = Leaddata['TotalVisits'].quantile(0.99)
Leaddata['Page Views Per Visit'].loc[Leaddata['Page Views Per Visit'] >= Leaddata['Page Views Per Visit'].quantile(0.99)] = Leaddata['Page Views Per Visit'].quantile(0.99)


# In[69]:


plt.figure(figsize = (10, 14))

plt.subplot(2,1,1)
sns.boxplot(Leaddata['TotalVisits'])

plt.subplot(2,1,2)
sns.boxplot(Leaddata['Page Views Per Visit'])
plt.show()


# It is clear that we were able to significantly reduce the number of outliers by capping at 99th percentile

# In[70]:


plt.figure(figsize = (20,20))

plt.subplot(3,3,1)
sns.countplot(x='TotalVisits', hue='Converted', data= Leaddata).tick_params(axis='x', rotation = 90)
plt.title('TotalVisits')


# In[71]:


plt.figure(figsize = (20,20))

plt.subplot(3,3,1)
sns.countplot(x='Page Views Per Visit', hue='Converted', data= Leaddata).tick_params(axis='x', rotation = 90)
plt.title('Page Views Per Visit')


# ### DATA PREPARATION

# ##### BINARY COLUMN

# In[72]:


#binary columns to 0 & 1
#We have 2 binary columns 'Do Not Email' & 'A free copy of Mastering The Interview' 

binlist = ['Do Not Email', 'A free copy of Mastering The Interview']

# Defining the function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function
Leaddata[binlist] = Leaddata[binlist].apply(binary_map)
Leaddata.head()


# ##### DUMMY VARIABLE 

# In[73]:


# Creating a dummy variable for categorical variables
dummy1 = pd.get_dummies(Leaddata[['Lead Origin', 'Lead Source', 'Specialization', 'What is your current occupation', 'City']], drop_first = True)

# Adding the results to the master dataframe
Leaddata = pd.concat([Leaddata, dummy1], axis=1)


# In[74]:


# Removing column whose dummy has been created

Leaddata.drop(['Lead Origin', 'Lead Source', 'Specialization', 'What is your current occupation', 'City'],axis=1,inplace=True)
Leaddata.head()


# ##### TEST-TRAIN SPLITTING

# In[75]:


#Importing train_test_split

from sklearn.model_selection import train_test_split


# In[76]:


#Assigning feature variable to X

X=Leaddata.drop(['Converted'],axis=1)
X.head()


# In[77]:


#Assigning response variable to Y

y=Leaddata['Converted']
y.head()


# In[78]:


# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.65, test_size=0.35, random_state=100)


# ##### FEATURE SCALING
# 

# In[79]:


# Import MinMax scaler
from sklearn.preprocessing import MinMaxScaler


# Scale the three numeric features
scaler = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])
X_train.head()


# ##### MODEL BUILDING
# 

# In[80]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
lr = LinearRegression()

# Running RFE with 14 variables as output
rfe = RFE(estimator=lr, n_features_to_select=14)
rfe.fit(X_train, y_train)


# In[81]:


# Features that have been selected by RFE

list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[82]:


# Put all the columns selected by RFE in the variable 'col'

col = X_train.columns[rfe.support_]


# In[83]:


# Selecting columns selected by RFE

X_train = X_train[col]


# In[84]:


# Importing statsmodels

import statsmodels.api as sm


# In[85]:


X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[86]:


# Importing 'variance_inflation_factor'

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[87]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# We will remove variable with High VIF value to bring max VIF under 5

# In[88]:


X_train.drop(['City_Not provided','Specialization_Not Provided'], axis = 1, inplace = True)


# In[89]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[90]:


# Make a VIF dataframe for the remaining variables

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Now as all VIF is under 5, we can proceed with model making

# ##### CREATING PREDICTION

# In[91]:


# Predicting the probabilities on the train set

y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[92]:


# Reshaping to an array

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[93]:


# Data frame with given convertion rate and probablity of predicted ones

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# In[94]:


# Substituting 0 or 1 with the cut off as 0.5
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# ##### EVALUATION OF MODEL

# In[95]:


# Importing metrics from sklearn for evaluation
from sklearn import metrics


# In[96]:


# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[97]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# In[98]:


# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]


# In[99]:


# Calculating the sensitivity
TP/(TP+FN)


# In[100]:


# Calculating the specificity
TN/(TN+FP)


# With cutoff of 0.5, we have accuracy of 78%, sensitivity of 62% and specificity of 88%

# ##### OPTIMISE CUT OFF

# In[101]:


# ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[102]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[103]:


# Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# #### Area Under ROC curve= 0.83,  And hence good
# 

# In[104]:


# Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[105]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[106]:


# Get the row with the highest Youden's J statistic
optimal_cutoff_row = cutoff_df.loc[(cutoff_df['sensi'] + cutoff_df['speci']-1).idxmax()]

# Retrieve the optimal cutoff value from the 'prob' column of the row
optimal_cutoff = optimal_cutoff_row['prob']

# Print the optimal cutoff value
print('Optimal cutoff:', optimal_cutoff)


# In[107]:


# Plotting the same
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[108]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.3 else 0)
y_train_pred_final.head()


# In[109]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[110]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[111]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[112]:


# Calculating the sensitivity
TP/(TP+FN)


# In[113]:


# Calculating the specificity
TN/(TN+FP)


# With cutoff as 0.3, the model seems to have accuracy of 78%, sensitivity of 77% and specificity of 79%

# ##### PREDICTION ON TEST SET

# In[114]:


#Scaling numeric values
X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[115]:


col = X_train.columns


# In[116]:


# Select the columns in X_train for X_test as well
X_test = X_test[col]
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test[col])
X_test_sm
X_test_sm


# In[117]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[118]:


# Making prediction using cut off 0.30
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.30 else 0)
y_pred_final


# In[119]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[120]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[121]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[122]:


# Calculating the sensitivity
TP/(TP+FN)


# In[123]:


# Calculating the specificity
TN/(TN+FP)


# On test set, with cutoff as 0.3, the model seems to have accuracy of 77%, sensitivity of 75% and specificity of 78%

# ##### PRECISION-RECALL

# In[124]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[125]:


# Precision = TP / TP + FP
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[126]:


#Recall = TP / TP + FN
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# With cutoff at 0.3, we are getting precision of 76% recall of 62%

# ##### PRECISION AND RECALL TRADEOFF

# In[127]:


from sklearn.metrics import precision_recall_curve


# In[128]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[129]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[130]:


#plotting the graph

plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[131]:


# Find the probability threshold that maximizes the F1 score
f1_scores = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1])
best_threshold = thresholds[np.argmax(f1_scores)]

print('Best probability threshold:', best_threshold)


# In[132]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.304 else 0)
y_train_pred_final.head()


# In[133]:


# Accuracy

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[134]:


# Creating confusion matrix again

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[135]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[136]:


# Precision = TP / TP + FP

TP / (TP + FP)


# In[137]:


#Recall = TP / TP + FN

TP / (TP + FN)


# With cutoff of 0.323 the model seems to have accuracy of 78%, precision of 69% and recall rate of 77% on training set

# ##### PREDICTION ON TEST SET

# In[138]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[139]:


# Making prediction using cut off 0.304

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.304 else 0)
y_pred_final


# ##### ACCURACY OVERALL

# In[140]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[141]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[142]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[143]:


# Precision = TP / TP + FP

TP / (TP + FP)


# In[144]:


#Recall = TP / TP + FN

TP / (TP + FN)


# With cutoff of 0.304 the model seems to have accuracy of 78%, Precision of about 71% and Recall about 75% on test set

# In[146]:


# Sorting variables that matters most in descending order 

logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()

# obtain the coefficient values and p-values for each predictor variable
coef_df = pd.DataFrame({'coef': result.params, 'p-value': result.pvalues})

# sort the coefficients by absolute value in descending order
coef_df['abs_coef'] = abs(coef_df['coef'])
coef_df = coef_df.sort_values(by=['abs_coef'], ascending=False)

# print the top predictors with the highest absolute coefficients in descending order
print("The variables that mattered the most in the potential buyers are (in descending order):")
for idx, row in coef_df.iterrows():
    print(idx, "| Coefficient:", round(row['coef'],2))


# #### Hence variables that mattered most in descending order are:
#     
# * Total Time Spent on Website
# * Lead Origin_Lead Add Form
# * Lead Origin_Lead Import
# * Page Views Per Visit
# * Do Not Email
# * Lead Source_Olark Chat
# * What is your current occupation_Working Professional
# * Lead Origin_Landing Page Submission
# * Lead Source_Reference
# * Specialization_Industry Specializations
# * A free copy of Mastering The Interview
# * TotalVisits

# ### USING THESE VARIABLE, THE X EDUCATION CAN DECIDE UPON THE AREA TO FOCUS AND MAY HELP THEM IN GETTING MORE CUSTOMERS
