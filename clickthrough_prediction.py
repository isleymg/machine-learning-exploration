
# coding: utf-8

# In[2]:


'''
Predicting Clickthrough Rates with Decision Trees
-------------------------------------------------
- Grow tree greedily by making series of local optimizations on choosing 
the most significant feature to partition on
- Split based on that feature
- Splitting process stops at subgroup where:
    - Num of samples is not greater than minimum # of samples for a new node
    - Maximum depth of the tree is reached

'''

import matplotlib.pyplot as plt
import numpy as np

'''
Gini Impurity: class impurity rate (lower means purer dataset)
A datset with only one class has Gini impurity of 0.
'''
pos_fraction = np.linspace(0.00, 1.00, 1000)
gini = 1 - pos_fraction**2 - (1-pos_fraction)**2
plt.plot(pos_fraction, gini)
plt.ylim(0, 1)
plt.xlabel('Positive fraction')
plt.ylabel('Gini Impurity')
plt.show()


# In[4]:


# Implementation of Gini impurity calculation function
def gini_impurity(labels):
    # When the set is empty, it is also pure
    if not labels:
        return 0
    # Count the occurrences of each label
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)

print('{0:.4f}'.format(gini_impurity([1, 1, 0, 1, 0])))


# In[6]:


'''
Entropy: probabilistic measure of uncertainty
Lower entropy implies a purer dataset with less ambiguity
Dataset with one class has entropy 0.
'''

pos_fraction = np.linspace(0.00, 1.00, 1000)
ent = - (pos_fraction * np.log2(pos_fraction) + (1 - pos_fraction) * np.log2(1 - pos_fraction))
plt.plot(pos_fraction, ent)
plt.ylim(0, 1)
plt.xlabel('Positive fraction')
plt.ylabel('Entropy')
plt.show()


# In[7]:


# Implementation of entropy calculation function
def entropy(labels):
    if not labels:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return - np.sum(fractions * np.log2(fractions))


# In[11]:


'''
Information Gain = Entropy(parent) - Entropy(children)
Want to maximize information gain from splitting on different features 
Both gini impurity and information gain measure the weighted impurity of children after a split
'''

criterion_function =  {'gini': gini_impurity,'entropy': entropy}
def weighted_impurity(groups, criterion='gini'):
    '''Calculate weighted impurity of children after split
    Args: 
        groups (list of children, children consist of a list of class labels)
        criterion (metric to measure quality of a split, 'gini' or 'entropy')
    Returns:
        float, weighted impurity
    '''
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * criterion_function[criterion](group)
    return weighted_sum

children_1 = [[1, 0, 1], [0, 1]]
children_2 = [[1, 1], [0, 0, 1]]
print('Entropy of #1 split: {0:.4f}'.format(weighted_impurity(children_1, 'entropy')))
print('Entropy of #2 split: {0:.4f}'.format(weighted_impurity(children_2, 'entropy')))


