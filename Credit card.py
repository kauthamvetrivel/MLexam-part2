#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install plotly


# In[1]:


import pandas as pd


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
warnings.filterwarnings(action="ignore")


# In[5]:


df = pd.read_csv('/Users/ks/Desktop/newfolder/credit_card.csv')


# In[6]:


df


# In[7]:


df.isnull().sum()


# In[8]:


cat_col = df.select_dtypes(include=['object']).columns
num_col = df.select_dtypes(exclude=['object']).columns


# In[9]:


sns.boxplot(x = 'TENURE', y = 'CREDIT_LIMIT', data = df,palette='Pastel1');


# In[10]:


df[num_col].corr()


# In[11]:


plt.subplots(figsize=(21,16))
sns.heatmap(df[num_col].corr(),annot = True);


# In[12]:


from sklearn.impute import KNNImputer
imputer = KNNImputer()
imp_data = pd.DataFrame(imputer.fit_transform(df[num_col]),columns=df[num_col].columns)
imp_data.isna().sum()


# In[13]:


imp_data


# In[14]:


pca = PCA()
pca.fit(imp_data)
PCA()
components = pca.transform(imp_data)
components = pd.DataFrame(components)
pcaRatio = pca.explained_variance_ratio_
pcaCS=np.cumsum(pcaRatio)


# In[15]:


px.line(pcaCS)


# In[16]:


pca = PCA(n_components = 5)


# In[17]:


pca.fit(imp_data)


# In[18]:


components = pca.transform(imp_data)
components = pd.DataFrame(components)


# In[19]:


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(components)
    distortions.append(kmeanModel.inertia_)


# In[20]:


model = KMeans(n_clusters=3, random_state=42).fit(components)


# In[21]:


score = silhouette_score(components, model.labels_, metric='euclidean')


# In[22]:


print('Silhouetter Score: %.3f' % score)


# In[ ]:





# In[ ]:





# In[ ]:




