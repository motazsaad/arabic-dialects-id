import glob
import os
import eval_metrics

from langid.langid import LanguageIdentifier, model

n = '5' 
arabic_dialect_model = open('Train_Padic_model_'+n+'_grams/model').read()
identifier = LanguageIdentifier.from_modelstring(arabic_dialect_model, norm_probs=True)

test_dir = 'test_padic/conversations/*.test' # how to specify the subdirectory
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
    accuracy, precision, recall, f_score = eval_metrics.evaluate_acc_PRF(prediction_list, y_list, base_name)
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('f-score: {}'.format(f_score))
    print('-------------------------------------')
