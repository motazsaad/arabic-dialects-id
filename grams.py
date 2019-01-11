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
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import SGDClassifier
from sklearn import metrics
#from sklearn.model_selection import GridSearchCV

import numpy as np

#print('Gothenburg Corpus -JL uni word ')


categories = ['jo', 'lb','pa', 'sy']
twenty_train = load_files('Filter_Pure/train/', encoding = 'utf-8',decode_error='ignore')
#twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
print('1. Loading data...')
print( twenty_train.target_names)
#print(len(twenty_train.data))
#print(len(twenty_train.filenames))
#print("\n".join(twenty_train.data[0].split("\n")[:3]))
#print(twenty_train.target_names[twenty_train.target[0]])
#for t in twenty_train.target[:10]:
#    print(twenty_train.target_names[t])


print('2. Tokenizing text with scikit-learn')
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(twenty_train.data)
#print(X_train_counts.shape)
#print(count_vect.vocabulary_.get(u'كيف'))

print('3 From occurrences to frequencies')
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

print('4. Training a classifier')
#clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

#print('\t4.1. predicte new documents')
"""
docs_new = ['انا مش عارف شو الموقف حاليا ', ' كل صالونات لبنان تتحدث عن رائحه فساد ', 'لإنو بطلت فارقه معو ']#['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
"""

print('5. Pipeline')
"""
analyzer : ‘word’, ‘char’, ‘char_wb’
    Whether the feature should be made of word or character n-grams. Option ‘char_wb’ 
    creates character n-grams only from text inside word boundaries;
ngram_range : tuple (min_n, max_n)
    The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
    All values of n such that min_n <= n <= max_n will be used.
""" 
text_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(2,5))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()), ])
    #SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))
text_clf.fit(twenty_train.data, twenty_train.target)
#print(text_clf.fit(twenty_train.data, twenty_train.target))

print('6. Evaluation of the performance on the test set')
twenty_test = load_files('Filter_Pure/test/',encoding='utf-8',decode_error='ignore')
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

print('\t8 document classification testing')

docs_new = ['انا مش عارف شو الموقف حاليا ', 'كل صالونات لبنان تتحدث عن رائحه فساد ', 'بعرفش هيا هيك شو اعملك ']
#X_new_counts = count_vect.transform(docs_new)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = text_clf.predict(docs_new)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))




   





