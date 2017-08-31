import nltk
import itertools


def document_features(document, selected_features):
    document_words = set(document)
    features = {}
    for word in selected_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


def prepare_text(text):
    return text.split()


def prepare_train_test(dataset, selection, max, split_indx):
    dataset = [(prepare_text(doc), label) for doc, label in dataset]
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












###################################################################################
# generates n-gram (g = num of grams)
# for example, if g=3, then the fuction will generate unigrams, bigrams, and tri-grams from the text.
def generate_ngrams(word_list, g):
    mygrams = []
    unigrams = [word for word in word_list]
    mygrams += unigrams
    for i in range(2, g + 1): mygrams += ngrams(word_list, i)
    return mygrams


###################################################################################

# generate n-gram features in the form (n-gram, True), i.e., binary feature. In other words, the n-gram exists
def generate_features(mygrams):
    feats = dict([(word, True) for word in mygrams])
    return feats


###################################################################################

# generate features for a doc from selected features grams (selected from a corpus)
# taks 2 parameters:
# 1. document feature grams
# 2. corpus selected feature grams
def build_features(doc_feat_grams, corpus_feat_grams):
    doc_grams = set(doc_feat_grams)
    feats = dict([(word, True) for word in doc_grams if word in corpus_feat_grams])
    return feats
