#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:49:19 2017

@author: xabuka
"""
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np

print('SVM Classifier')

twenty_train = load_files('Pure Corpus/Filter_Pure/train/', encoding = 'utf-8',decode_error='ignore')

print('1. Loading data...')
print( twenty_train.target_names)
print(len(twenty_train.data))
print(len(twenty_train.filenames))

text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='epsilon_insensitive', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            )),])
text_clf.fit(twenty_train.data, twenty_train.target) 


twenty_test = load_files('Pure Corpus/Filter_Pure/test/',encoding='utf-8',decode_error='ignore')
docs_test = twenty_test.data
 
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target)  )


print('7. Evaluation Metrics') 
print(metrics.classification_report(twenty_test.target, predicted,
     target_names=twenty_test.target_names)) 
#print(predicted)
print(metrics.confusion_matrix(twenty_test.target, predicted))

