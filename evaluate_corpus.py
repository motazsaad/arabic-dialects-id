import glob
import os
import numpy as np

from langid.langid import LanguageIdentifier, model
n = '4'
arabic_dialect_model = open('Corpus_model_'+n+'_grams/model').read()
identifier = LanguageIdentifier.from_modelstring(arabic_dialect_model, norm_probs=True)


def evaluation(predictions, y_list, label):
    # encoding
    pred_labels = np.asarray([1 if p == label else 0 for p in predictions])
    true_labels = np.asarray([1 if y == label else 0 for y in y_list])
    # print('pred_labels.shape:', pred_labels.shape)
    # print('true_labels.shape:', true_labels.shape)
    # print('len(predictions):', len(predictions))
    # print('len(y_list):', len(y_list))
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP, FP, TN, FN))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print('TP + TN + FP + FN = ', TP + TN + FP + FN)
    f_score = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, f_score

test_dir = 'test_Our_Corpus/*.test' # how to specify the subdirectory
test_files = glob.glob(test_dir)

for test in test_files:
    base_name, ext = os.path.basename(test).split('.')
    lines = open(test).read().splitlines()
    prediction_list = list()
    y_list = list()
    for line in lines:
        if line:
            dialect, conf = identifier.classify(line)
            prediction_list.append(dialect)
            y_list.append(base_name)
            #print(dialect, base_name)
    print('evaluating ', base_name)
    accuracy, precision, recall, f_score = evaluation(prediction_list, y_list, base_name)
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('f-score: {}'.format(f_score))
    print('-------------------------------------')
