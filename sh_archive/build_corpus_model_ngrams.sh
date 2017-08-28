#!/usr/bin/env bash
n=4
rm -rf Train_Our_Corpus_${n}_grams

python2.7 langid.py-master/langid/train/index.py Train_Our_Corpus/ -m Train_Our_Corpus_model_${n}_grams

python2.7 langid.py-master/langid/train/tokenize.py --max_order ${n} Train_Our_Corpus_model_${n}_grams

python2.7 langid.py-master/langid/train/DFfeatureselect.py --max_order ${n} Train_Our_Corpus_model_${n}_grams

python2.7 langid.py-master/langid/train/IGweight.py -d Train_Our_Corpus_model_${n}_grams

python2.7 langid.py-master/langid/train/IGweight.py -lb Train_Our_Corpus_model_${n}_grams

python2.7 langid.py-master/langid/train/LDfeatureselect.py Train_Our_Corpus_model_${n}_grams

python2.7 langid.py-master/langid/train/scanner.py Train_Our_Corpus_model_${n}_grams

python2.7 langid.py-master/langid/train/NBtrain.py Train_Our_Corpus_model_${n}_grams

