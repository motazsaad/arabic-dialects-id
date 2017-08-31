import nltk
import os
import glob
import feature_extraction
from nltk.metrics.scores import f_measure
import collections
import sys


train = list()
test = list()

train_dir = 'train_multidialect_arabic/conversations/'
labels = os.listdir(train_dir)
for label in labels:
    files = glob.glob(train_dir + label + "/*")
    for file in files:
        text = open(file).read().strip()
        train.append((text, label))

test_dir = "test_multidialect_arabic"
test_files = glob.glob(test_dir + "/*")
for file in test_files:
    base_name = os.path.basename(file)
    label, ext = os.path.splitext(base_name)
    lines = open(file).readlines()
    for line in lines:
        test.append((line.strip(), label))





# print(train[0][0])
# print(train[0][1])

# print(test[0][0])
# print(test[0][1])

all_data = train + test
indx = len(train)
train_set, test_set = feature_extraction.prepare_train_test(dataset=all_data, selection='gt', max=3, split_indx=indx)
#train_set, test_set = feature_extraction.prepare_train_test(dataset=all_data, selection='top', max=2000, split_indx=indx)

#print(train_set[0])

print('training ...')
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('training is done')

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

test_size = len(test_set)
print('classifying ....')
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
    sys.stdout.write("\r{} of {} classified".format(i, test_size))

print('\naccuracy:', nltk.classify.accuracy(classifier, test_set))
for label in labels:
    print('{} f-score: {}'.format(label, f_measure(refsets[label], testsets[label])))


classifier.show_most_informative_features(20)

