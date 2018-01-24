
# coding: utf-8

# In[12]:


'''
Exploration topic: Fetching raw newsgroups data, filtering content, and clustering into topic categories

'''

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups 
groups = fetch_20newsgroups()


# In[16]:


'''
Filtering
---------------------------------------------------------------
Apply filtering to the highest count words

'''

from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

def letters_only(astr):
    return astr.isalpha()


# In[26]:


# Print list of 500 words that have the highest counts
cv = CountVectorizer(stop_words="english", max_features=500)
groups = fetch_20newsgroups()
cleaned = []
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

for post in groups.data:
        cleaned.append(' '.join([
                             lemmatizer.lemmatize(word.lower())
                             for word in post.split()
                             if letters_only(word)
                             and word not in all_names]))
        
transformed = cv.fit_transform(cleaned)
print(cv.get_feature_names())


# In[14]:


sns.distplot(np.log(transformed.toarray().sum(axis=0)))
plt.xlabel('Log Count')
plt.ylabel('Frequency')
plt.title('Distribution Plot of 500 Word Counts')
plt.show()


# In[28]:


'''
# K-means Clustering: divide newsgroup dataset into k clusters
---------------------------------------------------------------
1) Assign each data point a cluster with the lowest distance.
2) Recalculate the center of the cluster as the mean of the cluster points coordinates.

'''

from sklearn.cluster import KMeans

km = KMeans(n_clusters=20)
km.fit(transformed)
labels = groups.target
plt.scatter(labels, km.labels_)
plt.xlabel('Newsgroup')
plt.ylabel('Cluster')
plt.show()


# In[32]:


'''
Topic modeling - Non-negative Matrix Factorization (NMF)
---------------------------------------------------------------
Factorizes a matrix into a product of two smaller matrices such as the three matrices have no negative values
'''

from sklearn.decomposition import NMF
nmf = NMF(n_components=100, random_state=43).fit(transformed)
for topic_idx, topic in enumerate(nmf.components_):
    label= '{}: '.format(topic_idx)
    print (label, " ".join([cv.get_feature_names()[i] for i in topic.argsort()[:-9:-1]]))

