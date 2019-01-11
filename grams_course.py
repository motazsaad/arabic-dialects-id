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

import numpy as np



"""
 Lnaguage categories = 'jo': Jordinain , 'lb': Lebanese ,'pa':Palestinian , 'sy':Syrian
"""
corpus = load_files('train_Pure_corpus/stories/', encoding = 'utf-8',decode_error='ignore')
print('1. Loading data...')
print( 'Langauges : ',corpus.target_names)
print('Total Number of files/sentences = ',len(corpus.data))
#print(len(corpus.filenames))
#print("\n".join(corpus.data[0].split("\n")[:3]))
#print(corpus.target_names[corpus.target[0]])
#for t in corpus.target[:10]:
#    print(corpus.target_names[t])


print('2. Tokenizing text with scikit-learn')
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(corpus.data)
#print(X_train_counts.shape)
#print(count_vect.vocabulary_.get(u'كيف'))

print('3 From occurrences to frequencies')
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

print('4. Training a classifier')
#clf = MultinomialNB().fit(X_train_tfidf, corpus.target)


print('5. Pipeline')
"""
analyzer : ‘word’, ‘char’, ‘char_wb’
    Whether the feature should be made of word or character n-grams. Option ‘char_wb’ 
    creates character n-grams only from text inside word boundaries;
ngram_range : tuple (min_n, max_n)
    The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
    All values of n such that min_n <= n <= max_n will be used.
""" 
#using NB classifier
text_clf = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(2,3))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()), ])
#using SVM classifier
text_clf_SDG = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(2,3))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)), ])

text_clf.fit(corpus.data, corpus.target)
text_clf_SDG.fit(corpus.data, corpus.target)
#print(text_clf.fit(corpus.data, corpus.target))

print('6. Evaluation of the performance on the test set')
corpus_test = load_files('test_Pure_corpus/',encoding='utf-8',decode_error='ignore')
#corpus_test = load_files('/Users/xabuka/PycharmProjects/arabic-dialects-langid/test_padic/split/',encoding = 'utf-8', decode_error = 'ignore')
#corpus_test = load_files('/Users/xabuka/PycharmProjects/arabic-dialects-langid/test_multidialect_arabic_full/conv/',encoding = 'utf-8', decode_error = 'ignore')
docs_test = corpus_test.data
predicted = text_clf.predict(docs_test)
predicted_SDG = text_clf_SDG.predict(docs_test)

print('Accuracy for NB = ',np.mean(predicted == corpus_test.target))
#print(corpus_test.target_names)
print('Accuracy for SDG = ',np.mean(predicted_SDG == corpus_test.target))





print('7. Evaluation Metrics') 
print('Evaluation of NB :')
print(metrics.classification_report(corpus_test.target, predicted,
     target_names=corpus_test.target_names)) 
#print(predicted)
#print(metrics.confusion_matrix(corpus_test.target, predicted, labels=[0, 1, 2,3]))
#  'precision', 'predicted', average, warn_for)
print('Evaluation of SDG :')
print(metrics.classification_report(corpus_test.target, predicted_SDG,
     target_names=corpus_test.target_names)) 
#print(metrics.confusion_matrix(corpus_test.target, predicted_SDG, labels=[0, 1, 2,3]))

print('8 document classification testing')

docs_new = ['انا مش عارف شو الموقف حاليا ', 'كل صالونات لبنان تتحدث عن رائحه فساد ', 'بعرفش هيا هيك شو اعملك ']
predicted = text_clf.predict(docs_new)
predicted_SDG = text_clf_SDG.predict(docs_new)
print('NB Test')
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, corpus.target_names[category]))
print('SDG Test')
for doc, category in zip(docs_new, predicted_SDG):
    print('%r => %s' % (doc, corpus.target_names[category]))




   





