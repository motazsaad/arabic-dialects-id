import nltk
import itertools
import process_arabic
from nltk.util import ngrams


def document_features(document, selected_features=None):
    document_words = set(document)
    features = {}
    if selected_features:
        for feature in selected_features:
            features['contains({})'.format(feature)] = (feature in document_words)
    else:
        for word in document_words:
            features['contains({})'.format(word)] = True
    return features


def generate_ngrams(word_list, n=2):
    mygrams = []
    unigrams = [word for word in word_list]
    for i in range(2,n+1):
        mygrams += ngrams(word_list, i)
    grams = ['_'.join(g) for g in mygrams]
    return unigrams + grams


def prepare_text(text, order):
    text = process_arabic.remove_diacritics(text)
    text = process_arabic.remove_punctuation(text)
    words = text.split()
    if order == 1:
        return words
    else:
        n_grams = generate_ngrams(words, order)
        return n_grams


def prepare_train_test(dataset, order, selection, max, split_indx):
    dataset = [(prepare_text(doc, order), label) for doc, label in dataset]
    print("dataset size: {}".format(len(dataset)))

    all_features = list(itertools.chain.from_iterable(doc for doc, label in dataset))
    #print('sample of all features:\n{}'.format(all_features[:3]))

    all_features_freq = nltk.FreqDist(all_features)
    #print('sample of all_features_freq:\n{}'.format(all_features_freq))

    if selection == 'top':
        selected_features = list(all_features_freq)[:max]
    elif selection == 'gt':
        selected_features = list([word for word, freq in all_features_freq.items() if freq > max])
    print("{} are selected from {}".format(len(selected_features), len(all_features)))
    #print('sample of selected_features:\n{}'.format(selected_features[:3]))

    print('generating features for documents ...')
    feature_sets = [(document_features(d, selected_features), c) for d, c in dataset]
    #print('sample of feature_sets:\n{}'.format(feature_sets[:3]))
    train_set, test_set = feature_sets[:split_indx], feature_sets[split_indx:]
    print('train size {}'.format(len(train_set)))
    print('test size {}'.format(len(test_set)))
    print('features are ready ...')
    return train_set, test_set


def prepare_train_data(dataset, order, selection, max):
    dataset = [(prepare_text(doc, order), label) for doc, label in dataset]
    print("dataset size: {}".format(len(dataset)))
    print('model order (n=): {}'.format(order))

    all_features = list(itertools.chain.from_iterable(doc for doc, label in dataset))
    all_features_len = len(all_features)
    all_features_freq = nltk.FreqDist(all_features)
    del all_features
    if selection == 'top':
        selected_features = list(all_features_freq)[:max]
    elif selection == 'gt':
        selected_features = list([word for word, freq in all_features_freq.items() if freq > max])
    print("{} are selected from {}".format(len(selected_features), all_features_len))
    del all_features_freq
    print('generating features for documents ...')
    feature_sets = [(document_features(d, selected_features), c) for d, c in dataset]
    print('features are ready ...')
    return feature_sets


def prepare_test_data(dataset, order):
    dataset = [(prepare_text(doc, order), label) for doc, label in dataset]
    print("dataset size: {}".format(len(dataset)))
    print('model order (n=): {}'.format(order))
    print('generating features for documents ...')
    feature_sets = [(document_features(d), c) for d, c in dataset]
    print('features are ready ...')
    return feature_sets







