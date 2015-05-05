#!/usr/bin/python

import sys
import pickle
import numpy
from time import time

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import grid_search
from sklearn.cluster import KMeans

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data


#gets the mean number of to, from, to_poi and from_poi emails, ignoring NaNs
def get_mean_emails(data_dict):
    a = numpy.mean([int(data_dict[name]['to_messages']) for name in data_dict if data_dict[name]['to_messages'] != "NaN"])
    b = numpy.mean([int(data_dict[name]['from_messages']) for name in data_dict if data_dict[name]['to_messages'] != "NaN"])
    c = numpy.mean([int(data_dict[name]['from_this_person_to_poi']) for name in data_dict if data_dict[name]['from_this_person_to_poi'] != "NaN"])
    d = numpy.mean([int(data_dict[name]['from_poi_to_this_person']) for name in data_dict if data_dict[name]['from_poi_to_this_person'] != "NaN"])
    e = numpy.mean([int(data_dict[name]['shared_receipt_with_poi']) for name in data_dict if data_dict[name]['shared_receipt_with_poi'] != "NaN"])
    return (a,b,c,d,e)


def replace_email_NaNs(data_dict, email_means):
    for person in data_dict.keys():
        if data_dict[person]['to_messages'] == "NaN":
            data_dict[person]['to_messages'] = email_means[0]
        if data_dict[person]['from_messages'] == "NaN":
            data_dict[person]['from_messages'] = email_means[1]
        if data_dict[person]['from_this_person_to_poi'] == "NaN":
            data_dict[person]['from_this_person_to_poi'] = email_means[2]
        if data_dict[person]['from_poi_to_this_person'] == "NaN":
            data_dict[person]['from_poi_to_this_person'] = email_means[3]
        if data_dict[person]['shared_receipt_with_poi'] == "NaN":
            data_dict[person]['shared_receipt_with_poi'] = email_means[4]

    return data_dict



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'shared_receipt_with_poi', 'bonus', 'exercised_stock_options',
                 'other', 'deferral_payments']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
#data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

###add POIs I found through research
"""
nameList = ['WHALLEY LAWRENCE G', 'LAVORATO JOHN J', 'BUY RICHARD B', \
            'BUTTS ROBERT H', 'HUGHES JAMES A', 'MORDAUNT KRISTINA M', \
            'ECHOLS JOHN B']

def addPOI (nameList, data_dict):
    for name in nameList:
        data_dict[name]['poi'] = True
    return data_dict

data_dict = addPOI(nameList, data_dict)
"""

### Task 3: Create new feature(s)
#create feature: portion of emails from poi, to poi

#need to handle 59 people with NaN values: use mean of columns
#mean_to_messages, mean_from_messages, mean_to_poi_messages, mean_from_poi_messages
email_means = get_mean_emails(data_dict)

#replace NaN values in email features with column means
#so we can run SVM and create numeric features without NaN issues
data_dict = replace_email_NaNs(data_dict, email_means)

#make new features
for person in data_dict:
    data_dict[person]['email_from_poi_rate'] = \
        float(data_dict[person]['from_poi_to_this_person']) / int(data_dict[person]['to_messages'])
    data_dict[person]['email_to_poi_rate'] = \
        float(data_dict[person]['from_this_person_to_poi']) / int(data_dict[person]['from_messages'])

#add new features
features_list.append('email_from_poi_rate')
features_list.append('email_to_poi_rate')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True, remove_all_zeroes=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)


#clf = GaussianNB()
#clf = tree.DecisionTreeClassifier()
#clf = AdaBoostClassifier()
#clf = RandomForestClassifier()
#clf = SVC(kernel = 'rbf', C=100, gamma=0.1)
clf = KNeighborsClassifier(n_neighbors=5, weights='distance')



#use to remove NaNs if necessary, and replace with column means
"""
from sklearn.preprocessing import Imputer
# missing_values is the value of your placeholder, strategy is if you'd like mean, median or mode, and axis=0 means it calculates the imputation based on the other feature values for that sample
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(features)
features = imp.transform(features)
"""

#parameter tuning, uncomment to run one
"""
params = {'n_neighbors': [4,6,8,10], 'weights':('uniform', 'distance')}
clf = grid_search.GridSearchCV(kn, params)
"""
"""
params = {'n_clusters': [4, 5, 6, 7, 8, 9, 10], 'init': ('k-means++', 'random')}
km = KMeans(n_init=10)
clf=grid_search.GridSearchCV(km, params)
"""
"""
params = {'C': (1, 10, 100, 1000), 'gamma': (0, 0.01, 0.1, 1)}
sv = SVC()
clf=grid_search.GridSearchCV(sv, params)
"""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)


##########
