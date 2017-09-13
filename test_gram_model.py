import nltk
import os
import glob
import feature_extraction
from nltk.metrics.scores import f_measure, precision, recall
import collections
import sys
import pickle
import argparse

parser = argparse.ArgumentParser(description='test word gram model')
parser.add_argument('-t', '--test-dir', type=str,
                    help='test directory.', required=True)
parser.add_argument('-m', '--model', type=str,
                    help='model file.', required=True)
parser.add_argument('-o', '--order', type=int,
                    help='n-gram', required=True)
parser.add_argument('-l', '--level', type=str,
                    help='n-gram level (word or chars)', required=True)


def load_model(model_name):
    with open(model_name, 'rb') as model_reader:
        classifier = pickle.load(model_reader)
        print('model {} loaded'.format(model_name))
        return classifier


def test_model(test_dir, model_name, n_gram, level):
    classifier = load_model(model_name)
    labels = classifier.labels()
    print('labels: {}'.format(labels))
    test = list()

    test_files = glob.glob(test_dir + "/*")
    for file in test_files:
        base_name = os.path.basename(file)
        label, ext = os.path.splitext(base_name)
        lines = open(file).readlines()
        for line in lines:
            test.append((line.strip(), label))

    test_set = feature_extraction.prepare_test_data(dataset=test, order=n_gram, level=level)
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
        print('{0} precision: {1:0.2f}'.format(label, precision(refsets[label], testsets[label])))
        print('{0} recall: {1:0.2f}'.format(label, recall(refsets[label], testsets[label])))
        print('{0} f-score: {1:0.2f}'.format(label, f_measure(refsets[label], testsets[label])))
        print('---------------------------------------')
    classifier.show_most_informative_features(40)


# python test_gram_model.py -t test_multidialect_arabic -o 4 -m char_gram_models/multidialect_model_4g -l char
# python test_gram_model.py -t test_multidialect_arabic -o 1 -m word_gram_models/multidialect_model_1g -l word

if __name__ == '__main__':
    args = parser.parse_args()
    test_dir = args.test_dir
    model_name = args.model
    n = args.order
    l = args.level
    test_model(test_dir=test_dir, model_name=model_name, n_gram=n, level=l)
