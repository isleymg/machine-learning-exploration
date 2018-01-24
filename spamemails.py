
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

def get_posterior(term_document_matrix, prior, likelihood):
    '''
    Compute posterior of testing samples based on prior and likelihood
    Args:
        term_document_matrix (sparse matrix)
        prior dictionary{class label: prior}
        likelihood dictionary {class label:conditional probability vector}
    Returns:
       dictionary {class label: posterior}
    '''
    num_docs = term_document_matrix.shape[0]
    posteriors = []
    for i in range(num_docs):
        posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index]) * count

        min_log_posterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                posterior[label] = float('inf')
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors





# In[6]:


'''
Small test of Naive Bayes functions
'''

emails_test = [
    '''Subject: flat screens
   hello ,
   please call or contact regarding the other flat screens
   requested .
   trisha tlapek - eb 3132 b
   michael sergeev - eb 3132 a
   also the sun blocker that was taken away from eb 3131 a .
   trisha should two monitors also michael .
   thanks
   kevin moore''',
   '''Subject: having problems in bed ? we can help !
   cialis allows men to enjoy a fully normal sex life without
   having to plan the sexual act .
   if we let things terrify us, life will not be worth living
   brevity is the soul of lingerie .
   suspicion always haunts the guilty mind .'''
]

cleaned_test = clean_text(emails_test)

term_docs_test = cv.transform(cleaned_test)
posterior = get_posterior(term_docs_test, prior, likelihood)
print(posterior)

# [{1: 0.0032745671008375999, 0: 0.99672543289916238}, {1: 0.99999847255388452, 0: 1.5274461154428757e-06}]
# first email: 99.67% change it is genuine (not spam)
# second email: 99.99% change it is spam


# In[7]:


'''
Testing Naive Bayes Classifier with downloaded data
----------------------------------------------------
'''

# Split dataset into 66% training (learning) and 33% testing (prediction)
# with random_state=42 for consistent training & testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, labels, test_size = 0.33, random_state=42)

# Retrain term frequency CountVectorizer to recompute prior and likelihood
term_docs_train = cv.fit_transform(X_train)
label_index = get_label_index(Y_train)
prior = get_prior(label_index)
likelihood = get_likelihood(term_docs_train, label_index, smoothing)

# Predict posterior of test set
term_docs_test = cv.transform(X_test)
posterior = get_posterior(term_docs_test, prior, likelihood)




# In[8]:


'''
Evaluating Model Performance
'''

correct = 0.0
for pred, actual in zip(posterior, Y_test):
    print(pred, actual)
    if actual == 1:
        if pred[1] >= 0.5:
            correct += 1

    elif pred[0] > 0.5:
        correct += 1


print('The accuracy on {0} testing samples is: {1:.1f}%'.format(len(Y_test), correct/len(Y_test)*100))
# The accuracy on 1707 testing samples is: 92.0%


# In[9]:


'''
If we wanted to do what we just did but with built in libraries we can use:
- MultinomialNB to initialize a model
- Train the model with fit()
- Use predict_proba() to obtain prediction results
- Acquire predicted class balues with predict()
- Measure accuracy with score() and sklearn.metrics performance metrics functions

Same thing, a LOT less lines! (and a lot more abstracted)

'''

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(term_docs_train, Y_train)
prediction_prob = clf.predict_proba(term_docs_test)

prediction_prob[0:10]
'''returns
array([[  1.00000000e+00,   2.12716600e-10],
       [  1.00000000e+00,   2.72887131e-75],
       [  6.34671963e-01,   3.65328037e-01],
       [  1.00000000e+00,   1.67181666e-12],
       [  1.00000000e+00,   4.15341124e-12],
       [  1.37860327e-04,   9.99862140e-01],
       [  0.00000000e+00,   1.00000000e+00],
       [  1.00000000e+00,   1.07066506e-18],
       [  1.00000000e+00,   2.02235745e-13],
       [  3.03193335e-01,   6.96806665e-01]])
'''

prediction = clf.predict(term_docs_test)
prediction[:10]

accuracy = clf.score(term_docs_test, Y_test)
print('The accuracy using MultinomialNB is: {0:.1f}%\n'.format(accuracy*100))
# The accuracy using MultinomialNB is: 92.0%

'''
Important metrics:
Precision: positive classification that are correct out of all classfied
Recall: positive classification that are correct out of all positives classified
F1: 2 * ( Precision * Recall) / ( Precision + Recall )

'''
from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print(report)

'''
The accuracy using MultinomialNB is: 92.0%

             precision    recall  f1-score   support

          0       0.96      0.92      0.94      1191
          1       0.84      0.92      0.87       516

avg / total       0.92      0.92      0.92      1707

'''


# In[10]:


'''
Model Tuning and Cross Validation
----------------------------------
k-fold cross-validation: original data randomly divided into k equal-sized subsets, class priortion is preserved
    Each of k subsets is retained as testing set for model
    k-1 subsets form the training set for model
    Average performance across all k trails is calculated for overall result
'''

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
k=10
k_fold = StratifiedKFold(n_splits = k)
cleaned_emails_np = np.array(cleaned_emails)
labels_np = np.array(labels)

max_features_option = [2000, 4000, 8000]
smoothing_factor_option = [0.5, 1.0, 1.5, 2.0]
fit_prior_option = [True, False]
auc_record = {}


for train_indices, test_indices in k_fold.split(cleaned_emails, labels):
    X_train, X_test = cleaned_emails_np[train_indices], cleaned_emails_np[test_indices]
    Y_train, Y_test = labels_np[train_indices], labels_np[test_indices]
    for max_features in max_features_option:
        if max_features not in auc_record:
            auc_record[max_features] = {}
            cv = CountVectorizer(stop_words="english", max_features=max_features)
            term_docs_train = cv.fit_transform(X_train)
            term_docs_test = cv.transform(X_test)
            for smoothing_factor in smoothing_factor_option:
                if smoothing_factor not in auc_record[max_features]:
                    auc_record[max_features][smoothing] = {}
                for fit_prior in fit_prior_option:
                    clf = MultinomialNB(alpha=smoothing, fit_prior=fit_prior)
                    clf.fit(term_docs_train, Y_train)
                    prediction_prob = clf.predict_proba(term_docs_test)
                    pos_prob = prediction_prob[:, 1]
                    auc = roc_auc_score(Y_test, pos_prob)
                    auc_record[max_features][smoothing][fit_prior] = auc + auc_record[max_features][smoothing].get(fit_prior, 0.0)


print('max features  smoothing  fit prior  auc'.format(max_features, smoothing, fit_prior, auc/k))

for max_features, max_feature_record in auc_record.items():
    for smoothing, smoothing_record in max_feature_record.items():
        for fit_prior, auc in smoothing_record.items():
            print('       {0}      {1}      {2}    {3:.4f}'.format(max_features, smoothing, fit_prior, auc/k))
