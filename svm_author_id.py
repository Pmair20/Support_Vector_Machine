#!/usr/bin/python

"""
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from tools.email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn import metrics

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
clf = SVC(kernel="linear")
t0 = time()
clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t1 = time()
predicted = clf.predict(features_test)
print ("Prediction time:", round(time()-t1, 3), "s")
print("Accuracy:",metrics.accuracy_score(labels_test, predicted))
print(predicted)