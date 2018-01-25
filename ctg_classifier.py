
# coding: utf-8

# In[13]:


'''
Classifier that helps obstetricians categorize cardiotocograms (CTGs)
----------------------------------------------------------------------
Using: SVM with RBF kernel
Dataset (n=2126): https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls 
    Features: fetal heart rate and uterine contraction 
    Label: fetal state class code (1=normal, 2=suspect, 3=pathologic) 
'''
import pandas as pd
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import timeit



df = pd.read_excel('CTG.xls', "Raw Data")

# Assign feature set (Col D to AL)
X = df.iloc[1:2126, 3:-2].values

# Assign label set (Col AN)
Y = df.iloc[1:2126, -1].values 

Counter(Y) # Counter({1.0: 1654, 2.0: 295, 3.0: 176})

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# Set RBF kernel
svc = SVC(kernel='rbf')
parameters = {'C': (100, 1e3, 1e4, 1e5), 'gamma': (1e-08, 1e-7, 1e-6, 1e-5)}

grid_search = GridSearchCV(svc, parameters, n_jobs=-1, cv=3)
start_time = timeit.default_timer()
grid_search.fit(X_train, Y_train)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))


# In[18]:


# Optimal set of parameters (optimal C)
grid_search.best_params_

# Best 3-fold averaged performace under optimal C
grid_search.best_score_

# Retrieve SVM model with optimal parameter and apply to unknown testing set
svc_best = grid_search.best_estimator_

# Performance Metrics
accuracy = svc_libsvm_best.score(X_test, Y_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
# The accuracy on testing set is: 77.9%

from sklearn.metrics import classification_report
prediction = svc_best.predict(X_test)
report = classification_report(Y_test, prediction)
print(report)

