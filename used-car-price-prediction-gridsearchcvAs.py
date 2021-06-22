
# In[ ]:





# In[ ]:





# In[7]:


from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
import warnings
warnings.filterwarnings('ignore')


# ### Data Load: Load used car prices into a dataframe

# In[8]:

df1 = pd.read_csv('train-data.csv')
df1.head()


# In[9]:


df1.shape


# In[10]:


df1.columns


# In[11]:


df1.isnull().sum()


# # Data Cleaning: Handle NA values

# ##### Drop features that are containing 70% data null

# In[12]:


# In our dataset we had only New_Price which contain above 70% data
df2 = df1.dropna(thresh=df1.shape[0]*0.7, how='all',axis=1)


# In[13]:


df2.head(2)


# In[14]:


df2.isnull().sum()


# In[15]:


df2.shape


# In[16]:


df3 = df2.dropna()
df3.isnull().sum()


# In[17]:


df3.shape


# # Feature Engineering: Handle Categorical Variable

# ##### Create a company using Name column

# In[18]:


company = [i.split()[0] for i in df3['Name']]
df3.insert(0, "Company", company)


# #### Converting string data to numeric form, i.e removing units etc.

# In[19]:


df3['Mileage'] = pd.to_numeric(df3['Mileage'].str.lower().str.split().str.get(0), errors='coerce')
df3['Engine'] = pd.to_numeric(df3['Engine'].str.lower().str.split().str.get(0), errors='coerce')
df3['Power'] = pd.to_numeric(df3['Power'].str.lower().str.split().str.get(0), errors='coerce')


# # Data Analysis with Visualisation

# #### Q.) Which company sells most Number of cars?

# In[20]:


df3['Company'].value_counts()


# In[21]:


company_count=df3['Name'].str.lower().str.split().str.get(0).to_frame()["Name"].value_counts()
plt.figure(figsize=(12,12))
plt.xlabel('Company Name')
plt.ylabel('No. of cars sold')
plt.title('Cars sold per company')
company_count.plot(kind='bar')
plt.show()


# #### Q.) Which fuel Type has lesser mileage?

# In[22]:


mileage = df3.groupby('Fuel_Type').Mileage.mean()
plt.xlabel("Fuel Type")
plt.xlabel("Mileage")
plt.title("Fuel Type vs Mileage")
mileage.plot(kind='bar')
plt.show()


# #### Q.) What is the number of cars originally purchased per year?

# In[23]:


purchased_car_per_year = df3['Year'].value_counts()
purchased_car_per_year.plot(kind='bar')
plt.xlabel("Year")
plt.ylabel("Paurchased Cars")
plt.title("Year vs Purchased cars")
plt.show()


# #### Q.) What is the average price per company?

# In[24]:


average_price = df3.groupby('Company').Price.mean()
average_price.plot(kind='bar')
plt.show()


# #### Q.) After driving for how much KMs people like to sell their cars?

# In[25]:


Company_Kilometers_Driven = df3.groupby('Company').Kilometers_Driven.mean()
Company_Kilometers_Driven.plot(kind='bar')
plt.xlabel("s")
plt.ylabel("s")
plt.title("Average Kilometeres vs Company")
plt.show()


# In[26]:


df3.head(2)


# # Again Feature Engineering

# #### Examine Name which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of name

# In[27]:


df3.Name = df3.Name.apply(lambda x: x.strip())
name_stats = df3['Name'].value_counts(ascending=False)
name_stats


# In[28]:


name_stats.head()


# In[29]:


name_stats.values.sum()


# In[ ]:


print 


# In[30]:


len(name_stats[name_stats>10])


# In[31]:


len(name_stats)


# In[32]:


len(name_stats[name_stats<=10])


# # Dimensionality Reduction

# #### Any Name having less than 10 data points should be tagged as "other" name. This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns

# In[33]:


name_stats_less_than_10 = name_stats[name_stats<=10]
name_stats_less_than_10


# In[34]:


len(df3.Name.unique())


# In[35]:


df3.Name


# In[36]:


df3.Name = df3.Name.apply(lambda x: 'other' if x in name_stats_less_than_10 else x)
len(df3.Name.unique())


# In[37]:


df3.head(10)


# # Outlier Removal Using Business Logic

# In[38]:


df3[df3.Mileage<5].head()


# #### We can see above Mileage is 0.0 so we can remove those data or replace with mean, We will use mean with list comprehension.

# In[39]:


mileage_mean = df3.Mileage.mean()


# In[40]:


df3['Mileage'] = [i  if i > 5 else float(int(mileage_mean)) for i in df3.Mileage]


# In[41]:


# Again check minimum mileage


# In[42]:


df3.describe()


# In[43]:


# Check Outliers in Kilometers_Driven
df3['Kilometers_Driven'].min()
df3['Kilometers_Driven'].max()


# In[44]:


df3.sort_values(by='Kilometers_Driven', ascending=False)


# #### Check above data points. We have 65Million Kilometers_Driven and It is not possible. So we will handle this outlier using mean of Kilometer_Driven

# In[45]:


kilometers_driven_mean = df3['Kilometers_Driven'].mean()
df3['Kilometers_Driven'] = [i  if i < 1000000 else float(int(kilometers_driven_mean)) for i in df3.Kilometers_Driven]


# In[46]:


# Maximum Driven
df3['Kilometers_Driven'].max()


# In[47]:


df3.isnull().sum()


# #### We can see above some empty values are there in power variable so we will replace with mean.

# In[48]:


df3['Power'] = df3['Power'].fillna(df3['Power'].mean())


# # Identifying Categorical Data: Nominal, Ordinal and Continuous

# In[49]:


print(df3['Company'].value_counts())
print(df3['Location'].value_counts())
print(df3['Fuel_Type'].value_counts())
print(df3['Transmission'].value_counts())
print(df3['Owner_Type'].value_counts())


# #### Check above results here we can split Nominal and Ordinal

# In[50]:


# Convert year variable in age
df3['Year'] = 2021-df3['Year']


# ### Use Label Encoding for Ordinal Variables

# In[51]:


df3.head()


# # Use On Hot Encoding for Categorical Variables

# In[52]:


df3.head(2)


# In[53]:


company_dummies = pd.get_dummies(df3['Company'])
name_dummies = pd.get_dummies(df3['Name'])
location_dummies = pd.get_dummies(df3['Location'])
fuel_type_dummies = pd.get_dummies(df3['Fuel_Type'])
transmission_dummies = pd.get_dummies(df3['Transmission'])
owner_type_dummies = pd.get_dummies(df3['Owner_Type'])


# In[54]:


df3.head()


# # Collecting dataframe in a Dataframe

# In[55]:


# Create new dataframe with numeric variables
features = df3[['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']]


# In[56]:


df4 = pd.concat([features, company_dummies, name_dummies, location_dummies, fuel_type_dummies, transmission_dummies, owner_type_dummies], axis=1)


# In[57]:


df4.shape


# In[58]:


df4.head()


# ## Build Model Now...

# In[59]:


x = df4.drop(['Price'], axis=1)
y = df4['Price']


# In[60]:


x.shape


# In[61]:


from sklearn.preprocessing import MinMaxScaler
x[['Kilometers_Driven', 'Mileage']] = MinMaxScaler().fit_transform(x[['Kilometers_Driven', 'Mileage']])        


# In[62]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[63]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(x_train, y_train)
knn.score(x_test, y_test)


# In[64]:


df4.head()


# # Use K Fold cross validation to measure accuracy of our KNN model

# In[65]:


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(KNeighborsRegressor(), x, y, cv=cv)


# #### We can see that in 5 iterations we get a score approximately 80% all the time. This is pretty good but we want to test few other algorithms for regression to see if we can get even better score. We will use GridSearchCV for this purpose

# # Find best model using GridSearchCV

# #### Based on above results we can say that KNN gives the best score. Hence we will use that.

# In[ ]:


# Fit regression model
regr = DecisionTreeRegressor(max_depth=2)
# regr_2 = DecisionTreeRegressor(max_depth=5)
regr.fit(x_train, y_train)


# In[ ]:





# # Test the model for few cars

# In[ ]:





# In[78]:

#
#def predict_price(year, kilometers_driven, mileage, engine, power, seats, company, name, location, fuel_type, transmission, owner_type):
#    company_index = np.where(x.columns==company)[0][0]
#    name_index = np.where(x.columns==name)[0][0]
#    location_index = np.where(x.columns==location)[0][0]
#    fuel_type_index = np.where(x.columns==fuel_type)[0][0]
#    transmission_index = np.where(x.columns==transmission)[0][0]
#    owner_type_index = np.where(x.columns==owner_type)[0][0]
#    x2 = np.zeros(len(x.columns))
#    x2[0] = year
#    x2[1] = kilometers_driven
#    x2[2] = mileage
#    x2[3] = engine
#    x2[4] = power
#    x2[5] = seats
#    if company_index >= 0:
#        x2[company_index] = 1
#    if name_index >= 0:
#        x2[name_index] = 1
#    if location_index >= 0:
#        x2[location_index] = 1
#    if fuel_type_index >= 0:
#        x2[fuel_type_index] = 1
#    if transmission_index >= 0:
#        x2[transmission_index] = 1
#    if owner_type_index >= 0:
#        x2[owner_type_index] = 1
#
#    result = ([x2])
#    return result


# In[69]:
#data_unseen = pd.DataFrame([6, 0.052268, 0.324245, 1968, 140.80, 5.0, 'Audi', 'other', 'Coimbatore', 'Diesel', 'Automatic', 'Second'])
#
##Audi
##predict_price(6, 0.052268, 0.324245, 1968, 140.80, 5.0, 'Audi', 'other', 'Coimbatore', 'Diesel', 'Automatic', 'Second')
#
#
## In[72]:
#
#
## For Hundai
#ab=predict_price(6, 0.052268, 0.424245, 1582, 126.20, 5.0, 'Hyundai', 'Hyundai Creta 1.6 CRDi SX Option', 'Pune', 'Petrol', 'Manual', 'First')
#
#
## In[73]:
#
#
## For Maruti Wagon R
#predict_price(11, 0.092703, 0.744289, 998, 58.16, 5.0, 'Maruti', 'Maruti Wagon R LXI CNG', 'Mumbai', 'Petrol', 'Manual', 'First')
#

# In[74]:


# We can see below parameters
x.head()


# ## Export the tested model to a pickle file

# In[87]:


import pickle
pickle.dump(knn, open('model.pkl','wb'))

# In[88]:



# ## Export Categprical columns to a file that will be useful later on in our prediction application

# In[ ]:


import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# # Live Demo
# https://used-car-valuation-app.herokuapp.com/

# # Thank You
