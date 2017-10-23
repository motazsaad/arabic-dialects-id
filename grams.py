#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:19:33 2017

@author: xabuka
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:52:21 2017

@author: xabuka
"""
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import numpy as np

print('Gothenburg Corpus -JL uni word ')


categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = load_files('Pure Corpus/Filter_Pure/train/', encoding = 'utf-8',decode_error='ignore')

print('1. Loading data...')
print( twenty_train.target_names)
print(len(twenty_train.data))
print(len(twenty_train.filenames))
#print("\n".join(twenty_train.data[0].split("\n")[:3]))
#print(twenty_train.target_names[twenty_train.target[0]])
#for t in twenty_train.target[:10]:
#    print(twenty_train.target_names[t])


print('2. Tokenizing text with scikit-learn')
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(twenty_train.data)
#print(X_train_counts.shape)
#print(count_vect.vocabulary_.get(u'algorithm'))

print('3 From occurrences to frequencies')
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

print('4. Training a classifier')
#clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

print('4.1. predicte new documents')
"""
docs_new = ['انا مش عارف شو الموقف حاليا ', ' كل صالونات لبنان تتحدث عن رائحه فساد ', 'لإنو بطلت فارقه معو ']#['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
"""

print('5. Pipeline')
text_clf = Pipeline([('vect', CountVectorizer(analyzer='char_wb', ngram_range=(7,7))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()), ])
text_clf.fit(twenty_train.data, twenty_train.target)
#print(text_clf.fit(twenty_train.data, twenty_train.target))

print('6. Evaluation of the performance on the test set')
twenty_test = load_files('Pure Corpus/Filter_Pure/test/',encoding='utf-8',decode_error='ignore')
#twenty_test = load_files('/Users/xabuka/PycharmProjects/arabic-dialects-langid/test_padic/split/',encoding = 'utf-8', decode_error = 'ignore')
#twenty_test = load_files('/Users/xabuka/PycharmProjects/arabic-dialects-langid/test_multidialect_arabic_full/conv/',encoding = 'utf-8', decode_error = 'ignore')
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print('Accuracy = ',np.mean(predicted == twenty_test.target))
print(twenty_test.target_names)



print('7. Evaluation Metrics') 
print(metrics.classification_report(twenty_test.target, predicted,
     target_names=twenty_test.target_names)) 
#print(predicted)
print(metrics.confusion_matrix(twenty_test.target, predicted, labels=[0, 1, 2,3]))
#  'precision', 'predicted', average, warn_for)

"""

print('8. Parameter tuning using grid search')
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3), }
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
print(twenty_train.target_names[gs_clf.predict(['لإنو بطلت فارقه معو'])[0]])
print('best_score ',gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

"""



   





