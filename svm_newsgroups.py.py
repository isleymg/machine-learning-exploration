
# coding: utf-8

# In[16]:


'''
Support Vector Machines
are a set of supervised learning methods used for classification,
regression and outliers detection.

Three Scenarios:
1. Identifying the separating hyperplane in linearly separable data
    Any data point from one class satisfies wx + b > 0
    Any data point from another class satisfies wx + b < 0
2. Determining the optimal hyperplane
    Nearest point on positive side = positive hyperplane, wx^p + b = 1, x^p is point on positive hyperplane
    Nearest point on negative side = negative hyperplane, wx^n + b = -1, x^n is point on negative hyperplane
3. Handling outliers
    Allow misclassification of outliers and try to minimize error
4. Multiclasses
    One-vs-rest (k classifiers)
        For k classes, construct k different binary SVM classifiers
        For kth class, treat it as positive case and rest k-1 classes as negative
    One-vs-one (k(k-1)/2 classifiers)
        Conduct pairwise comparison by building SVM classifiers distinguishing data from each pair of classes

'''
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups

categories = [
    'comp.graphics',
    'sci.space',
    'alt.atheism',
    'talk.religion.misc',
    'rec.sport.hockey'
]
data_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# Clean data
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

cleaned_train = clean_text(data_train.data)
label_train = data_train.target
cleaned_test = clean_text(data_test.data)
label_test = data_test.target
len(label_train), len(label_test)


# In[19]:


# Check whether classes are imbalanced

from collections import Counter
Counter(label_train)  # Counter({0: 584, 1: 593})
Counter(label_test)   # Counter({0: 389, 1: 394})

# Extract tf-idf features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', max_features=8000)
term_docs_train = tfidf_vectorizer.fit_transform(cleaned_train)
term_docs_test = tfidf_vectorizer.transform(cleaned_test)

# Apply SVM algorithm
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Fit model to training set
svm.fit(term_docs_train, label_train)

# Predict on the testing set with trained model
# SVC handles multi-class (scenario #4 with one-vs-one scheme by default)
accuracy = svm.score(term_docs_test, label_test)
print('The accuracy on testing set is: {0:.1f}% \n'.format(accuracy*100))
# The accuracy on testing set is: 88.6%

from sklearn.metrics import classification_report
prediction = svm.predict(term_docs_test)
report = classification_report(label_test, prediction)
print(report)


# In[21]:


'''
SVM Kernels
-----------
Solves nonlinear classification problems by converting original feature space
to higher dimensional feature space x^i with a transformation function Φ
such that the transformed dataset Φ(x^i) is linearly separable

Kernel function: K(x^i, x^j) = x^i dot x^j

Radial basis function (RBF) aka Gaussian kernel is most popular

'''


categories = None
data_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

cleaned_train = clean_text(data_train.data)
label_train = data_train.target
cleaned_test = clean_text(data_test.data)
label_test = data_test.target

term_docs_train = tfidf_vectorizer.fit_transform(cleaned_train)
term_docs_test = tfidf_vectorizer.transform(cleaned_test)

# Linear kernel is good for classifying text data
svc_libsvm = SVC(kernel='linear')

# GridSearchCV handles data splitting, folds generation, cross training
# and validation, and exhaustive search over the best set of parameters
parameters = {'C': (0.1, 1, 10, 100)}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(svc_libsvm, parameters, n_jobs=-1, cv=3)


import timeit
start_time = timeit.default_timer()
grid_search.fit(term_docs_train, label_train)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))


# In[22]:


# Optimal set of parameters (optimal C)
grid_search.best_params_

# Best 3-fold averaged performace under optimal C
grid_search.best_score_

# Retrieve SVM model with optimal parameter and apply to unknown testing set
svc_libsvm_best = grid_search.best_estimator_
accuracy = svc_libsvm_best.score(term_docs_test, label_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
# The accuracy on testing set is: 76.2%

# In[23]:


# Using LinearSVC
from sklearn.svm import LinearSVC
svc_linear = LinearSVC()
grid_search = GridSearchCV(svc_linear, parameters, n_jobs=-1, cv=3)

start_time = timeit.default_timer()
grid_search.fit(term_docs_train, label_train)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))


# In[24]:


# Optimal set of parameters (optimal C)
grid_search.best_params_

# Best 3-fold averaged performace under optimal C
grid_search.best_score_

# Retrieve SVM model with optimal parameter and apply to unknown testing set
svc_libsvm_best = grid_search.best_estimator_
accuracy = svc_libsvm_best.score(term_docs_test, label_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
# The accuracy on testing set is: 77.9%

# LinearSVC outperforms livsvm because LinearSVC uses the liblinear library, designed for large datasets


# In[32]:


# Feature extraction and classification should be cross validated collectively
# Use Pipeline to do this
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('svc', LinearSVC()),
])

parameters_pipeline = {
    'tfidf__max_df': (0.25, 0.5),
    'tfidf__max_features': (40000, 50000),
    'tfidf__sublinear_tf': (True, False),
    'tfidf__smooth_idf': (True, False),
    'svc__C': (0.1, 1, 10, 100),
}

'''
Tuning the tdidf feature extractor:
    max_df: max doc frequency of a term to be allowed
    max_features: number of top features to consider
    sublinear_tf: scaling term freq with log function or not
    smoothing_idf: add initial 1 to doc freq
'''

grid_search = GridSearchCV(pipeline, parameters_pipeline, n_jobs=-1, cv=3)
start_time = timeit.default_timer()
grid_search.fit(cleaned_train, label_train)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))


# In[34]:


# Optimal set of parameters (optimal C)
grid_search.best_params_

# Best 3-fold averaged performace under optimal C
grid_search.best_score_

# Retrieve SVM model with optimal parameter and apply to unknown testing set
pipeline_best = grid_search.best_estimator_
accuracy = pipeline_best.score(cleaned_test, label_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
# The accuracy on testing set is: 80.6%
# (best accuracy)
