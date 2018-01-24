
# coding: utf-8

# In[1]:


'''
Exploration topic: Classify emails as SPAM or NOT SPAM with Naive Bayes supervised learning

Definitions:
------------
Bayes: It maps the probabilities of observing input features given belonging classes, to the probability distribution over classes based on Bayes' theorem. We will explain Bayes' theorem by examples in the next section.
Naive: It simplifies probability computations by assuming that predictive features are mutually independent


Data set:
---------
Directly downloaded compressed file from
http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz

Spam emails: 1500
Non-spam emails: 3672

'''
file_path = 'enron1/ham/0007.1999-12-14.farmer.ham.txt'
with open(file_path, 'r') as infile:
    ham_sample = infile.read()

print(ham_sample)


# In[9]:


'''
Import all necessary text files
'''
import glob 
import os
emails, labels = [], []

spam_file_path = 'enron1/spam'
for filename in glob.glob(os.path.join(spam_file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)
        
        
nonspam_file_path = 'enron1/ham'
for filename in glob.glob(os.path.join(nonspam_file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)
        
        
print (len(emails))
print (len(labels))


# In[13]:


'''
Preprocess and clean raw data
- Number and punctuation removal
- Human name removal (optional)
- Stop words removal
- Lemmatization
'''

from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
def letters_only(astr):
    return astr.isalpha()

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()


def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(
            ' '.join([lemmatizer.lemmatize(word.lower())
            for word in doc.split() if letters_only(word) and word not in all_names]))
    return cleaned_docs

cleaned_emails = clean_text(emails)
cleaned_emails[0]


# In[25]:


'''
Vectorize:
----------
Takes the document matrix (rows of words) into a term document matrix 
where each row is a term frequency sparse vector for a document and an email

Column format: (row index, feature/term index)
'''
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(stop_words="english", max_features=500) # considers 500 most frequent terms

term_docs = cv.fit_transform(cleaned_emails)
print(term_docs[0])

feature_names = cv.get_feature_names()
feature_names[357]


# term feature as key and feature index as value (example-- 'energy': 125)
feature_mapping = cv.vocabulary_
print(feature_mapping)


# In[38]:


'''
Build and Train Naive Bayes Model
---------------------------------
'''

def get_label_index(labels):
    '''Group data by label'''
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

label_index = get_label_index(labels)

def get_prior(label_index):
    '''Compute prior based on training samples
    Args: label_index
    Returns: dictionary {class_label: prior}
    '''
    prior = {label: len(index) for label, index in label_index.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior

prior = get_prior(label_index) # {1: 0.2900232018561485, 0: 0.7099767981438515}

import numpy as np
def get_likelihood(term_document_matrix, label_index, smoothing=0):
    '''Compute likelihood based on training samples
    Args: 
        term_document_matrix
        label_index
        smoothing
    Returns: 
        dictionary {class: conditional_prob(feature|class)}
    '''
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label]/float(total_count)
    return likelihood
     
    
smoothing = 1
likelihood = get_likelihood(term_docs, label_index, smoothing)

# first five elements of the conditional probability P(feature | spam) vector:
print(likelihood[0][:5]) # [ 0.00108581  0.00095774  0.00087978  0.00084637  0.00010023]

# corresponding terms
print(feature_names[:5]) # ['able', 'access', 'account', 'accounting', 'act']
                                       

