#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Learning Approaches

# #### Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from itertools import product
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from yellowbrick.cluster import KElbowVisualizer


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


#Reading the Dataset
df = pd.read_csv('Dataset-C-TelcoChurn.csv')


# In[4]:


#Visualize first 5 rows of the data
df.head()


# In[5]:


#Get dimension of the DataFrame
df.shape


# In[6]:


#Print information of the data (eg: indexes , non-null values)
df.info()


# In[7]:


#Print data types of each columns
df.dtypes


# In[8]:


#Drop unneeded columns
df = df.drop('customerID', axis=1)


# In[9]:


#Change data type to numerical for continuous variable
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[10]:


#Check null counts
df.isnull().sum() 


# In[11]:


#Replace null value with 0
df['TotalCharges'].fillna(0, inplace=True)


# In[12]:


#Re-verify null counts
df.isnull().sum() 


# In[13]:


#Visualize the proportions of Churn
plt.figure(figsize = (15,5))
df['Churn'].value_counts().plot(kind = 'pie',autopct = '%.2f', explode = [0,0.1])


# In[14]:


#Visualize the countplots for all categories against Churn
cat_col = df.select_dtypes(include = ['object'])
cat_col.drop(columns = ['Churn'],inplace = True)
j=1 

plt.figure(figsize = (15,25))

for i in cat_col.columns:
    plt.subplot(8,2,j)
    sns.countplot(x = df[i],hue = df['Churn'])
    j+=1

plt.tight_layout()


# In[15]:


#Statistical overview of the continuous variables in the dataset
df.describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu')


# In[16]:


#Overview of the categorical features in the dataset
df.describe(include=object).T


# In[17]:


#Getting the value counts in gender column
df['gender'].value_counts()


# In[18]:


#Label encoding the values to indicate binary values
df['gender'] = df['gender'].replace('Male', 1)
df['gender'] = df['gender'].replace('Female', 0)
df['gender'].value_counts()


# In[19]:


#Getting the value counts in Partner column
df['Partner'].value_counts()


# In[20]:


#Label encoding the values to indicate binary values
df['Partner'] = df['Partner'].replace('Yes', 1)
df['Partner'] = df['Partner'].replace('No', 0)
df['Partner'].value_counts()


# In[21]:


#Getting the value counts in Dependents column
df['Dependents'].value_counts()


# In[22]:


#Label encoding the values to indicate binary values
df['Dependents'] = df['Dependents'].replace('Yes', 1)
df['Dependents'] = df['Dependents'].replace('No', 0)
df['Dependents'].value_counts()


# In[23]:


#Getting the value counts in Phone Service column
df['PhoneService'].value_counts()


# In[24]:


#Label encoding the values to indicate binary values
df['PhoneService'] = df['PhoneService'].replace('Yes', 1)
df['PhoneService'] = df['PhoneService'].replace('No', 0)
df['PhoneService'].value_counts()


# In[25]:


#Getting the value counts in Phone Service column
df['PaperlessBilling'].value_counts()


# In[26]:


#Label encoding the values to indicate binary values
df['PaperlessBilling'] = df['PaperlessBilling'].replace('Yes', 1)
df['PaperlessBilling'] = df['PaperlessBilling'].replace('No', 0)
df['PaperlessBilling'].value_counts()


# In[27]:


#Getting the value counts in Churn column
df['Churn'].value_counts()


# In[28]:


#Label encoding the values to indicate binary values
df['Churn'] = df['Churn'].replace('Yes', 1)
df['Churn'] = df['Churn'].replace('No', 0)
df['Churn'].value_counts()


# ### For the remaining categorical data, one-hot encoding method is chosen to convert it to a numerical value. 
# ### Ordinal and label encoding would not be appropriate as it would introduce ranking in the data

# In[29]:


#Create a second dataframe and generate the one-hot encoding for all remaining categorical data
df2 = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'])


# In[30]:


#Overview of the one-hot encoded data
df2.info()


# # K-Means (Partition-Based)

# ### In order to find an appropriate number of clusters, the elbow method will be used. In this method for this case, the inertia for a number of clusters between 2 and 10 will be calculated. The rule is to choose the number of clusters where you see a kink or "an elbow" in the graph.

# In[31]:


#Visualizing the elbow method
X = df2.values[:,:]
model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2,10))

visualizer.fit(X)
visualizer.show()
plt.show()


# ### The graph above shows the reduction of a distortion score as the number of clusters increases. However, there is no clear "elbow" visible. The underlying algorithm suggests 4 clusters. A choice of 4 or 5 clusters seems to be fair. Another way to choose the best number of clusters is to plot the silhuette score in a function of number of clusters. Let's see the results.

# In[32]:


#Implementing silhoutte coefficient in the elbow chart
model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

visualizer.fit(X)
visualizer.show()
plt.show()


# In[33]:


#Fit the algorithm using 2 clusters as suggested
#Assign the k-means label in the dataframe

KM_clusters = KMeans(n_clusters=2, init='k-means++').fit(X) 
labels = KM_clusters.labels_
df2['KM_Clus'] = labels


# In[34]:


#Create a new dataframe to get the relationship between the K-mean labels and mean of the data
df3 = df2.groupby('KM_Clus').mean()
df3


# In[35]:


#Visualizing the clusters in Tenure vs Monthly Charges
plt.scatter(df2.iloc[:, 4], df2.iloc[:, 7], c=labels.astype(np.float), alpha=0.5, cmap='viridis')
plt.xlabel('Tenure', fontsize=18)
plt.ylabel('Monthly Charges', fontsize=16)

plt.show()


# In[36]:


#Visualizing 3d plot of Tenure, Monthly Charges and Churn
import plotly as py
import plotly.graph_objs as go

def tracer(db, n, name):
    '''
    This function returns trace object for Plotly
    '''
    return go.Scatter3d(
        x = db[db['KM_Clus']==n]['tenure'],
        y = db[db['KM_Clus']==n]['MonthlyCharges'],
        z = db[db['KM_Clus']==n]['Churn'],
        mode = 'markers',
        name = name,
        marker = dict(
            size = 5
        )
     )

trace0 = tracer(df2, 0, 'Cluster 0')
trace1 = tracer(df2, 1, 'Cluster 1')
trace2 = tracer(df2, 2, 'Cluster 2')
trace3 = tracer(df2, 3, 'Cluster 3')
trace4 = tracer(df2, 4, 'Cluster 4')

data = [trace0, trace1, trace2, trace3, trace4]

layout = go.Layout(
    title = 'Clusters by K-Means',
    scene = dict(
            xaxis = dict(title = 'Tenure'),
            yaxis = dict(title = 'Monthly Charges'),
            zaxis = dict(title = 'Churn')
        )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# In[37]:


#Check the quality of each cluster

from yellowbrick.cluster import SilhouetteVisualizer
model = KMeans(n_clusters=2, random_state=0)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(X)
visualizer.show()
plt.show()


# # DBSCAN (Density-Based)

# ### It is difficult arbitrarily to say what values of epsilon and min_samples will work the best. Therefore, first a matrix of investigated combinations shall be created.

# In[38]:


eps_values = np.arange(8,12.75,0.25) # eps values to be investigated
min_samples = np.arange(3,10) # min_samples values to be investigated

DBSCAN_params = list(product(eps_values, min_samples))


# ### Because DBSCAN creates clusters itself based on those two parameters let's check the number of generated clusters.

# In[39]:


no_of_clusters = []
sil_score = []

for p in DBSCAN_params:
    DBS_clustering = DBSCAN(eps=p[0], min_samples=p[1]).fit(X)
    no_of_clusters.append(len(np.unique(DBS_clustering.labels_)))
    sil_score.append(silhouette_score(X, DBS_clustering.labels_))


# ### A heatplot below shows how many clusters were generated by the DBSCAN algorithm for the respective parameters combinations.

# In[40]:


tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
tmp['No_of_clusters'] = no_of_clusters

pivot_1 = pd.pivot_table(tmp, values='No_of_clusters', index='Min_samples', columns='Eps')

fig, ax = plt.subplots(figsize=(15,6))
sns.heatmap(pivot_1, annot=True,annot_kws={"size": 8}, cmap="YlGnBu", ax=ax)
ax.set_title('Number of clusters')
plt.tight_layout()
plt.show()


# ### The heatplot above shows, the number of clusters varies greatly. To decide which combination to choose, a metric - a silhuette score will be plotted as a heatmap again.

# In[41]:


tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
tmp['Sil_score'] = sil_score

pivot_1 = pd.pivot_table(tmp, values='Sil_score', index='Min_samples', columns='Eps')

fig, ax = plt.subplots(figsize=(18,6))
sns.heatmap(pivot_1, annot=True, annot_kws={"size": 10}, cmap="YlGnBu", ax=ax)
plt.tight_layout()
plt.show()


# ### Values are in negative, indicating something is wrong with the data or the algorithm. To check whether is there something wrong with the data, the model are trained again using only continuous variables. All binary data and one-hot encoded data are omitted. The above processes for DBSCAN are repeated.

# In[42]:


df4 = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
X = df4.values[:,:]

no_of_clusters = []
sil_score = []

for p in DBSCAN_params:
    DBS_clustering = DBSCAN(eps=p[0], min_samples=p[1]).fit(X)
    no_of_clusters.append(len(np.unique(DBS_clustering.labels_)))
    sil_score.append(silhouette_score(X, DBS_clustering.labels_))


# In[43]:


#Visualize the number of clusters for the updated criterias
tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
tmp['No_of_clusters'] = no_of_clusters

pivot_1 = pd.pivot_table(tmp, values='No_of_clusters', index='Min_samples', columns='Eps')

fig, ax = plt.subplots(figsize=(15,6))
sns.heatmap(pivot_1, annot=True,annot_kws={"size": 8}, cmap="YlGnBu", ax=ax)
ax.set_title('Number of clusters')
plt.tight_layout()
plt.show()


# In[44]:


#Visualize the heatmap of silhuette scores for the updated criteria
tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
tmp['Sil_score'] = sil_score

pivot_1 = pd.pivot_table(tmp, values='Sil_score', index='Min_samples', columns='Eps')

fig, ax = plt.subplots(figsize=(18,6))
sns.heatmap(pivot_1, annot=True, annot_kws={"size": 10}, cmap="YlGnBu", ax=ax)
plt.tight_layout()
plt.show()


# ### The negative values of the updated silhuette scores are indicative that the dataset is not suitable for DBSCAN as no apparent number of clusters can be generated. To visualize this, a scatterplot of Tenure vs Monthly Charges shall show that there are no apparent clusters that can be derived using density-based cluster methods unlike partition-based clustering methods where number of clusters are specified by the user 

# In[45]:


#Scatterplot of Tenure vs Monthly Charges

plt.scatter(df4['tenure'], df4['MonthlyCharges'])
plt.xlabel('Tenure', fontsize=18)
plt.ylabel('Monthly Charges', fontsize=16)
plt.show()


# In[ ]:




