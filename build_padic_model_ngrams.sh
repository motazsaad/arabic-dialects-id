#!/usr/bin/env bash
n=4
rm -rf padic_model_${n}_grams

python langid.py-master/langid/train/index.py train_our_corpus/ -m padic_model_${n}_grams

python langid.py-master/langid/train/tokenize.py --max_order ${n} padic_model_${n}_grams

python langid.py-master/langid/train/DFfeatureselect.py --max_order ${n} padic_model_${n}_grams

python langid.py-master/langid/train/IGweight.py -d padic_model_${n}_grams

python langid.py-master/langid/train/IGweight.py -lb padic_model_${n}_grams

python langid.py-master/langid/train/LDfeatureselect.py padic_model_${n}_grams

python langid.py-master/langid/train/scanner.py padic_model_${n}_grams

python langid.py-master/langid/train/NBtrain.py padic_model_${n}_grams

#python langid.py -m padic_model_4_grams/model