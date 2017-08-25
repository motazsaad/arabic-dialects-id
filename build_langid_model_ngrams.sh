#!/usr/bin/env bash

n=6
rm -rf multidialect_model_${n}_grams

python langid.py-master/langid/train/index.py train_multidialect_arabic/ -m multidialect_model_${n}_grams

python langid.py-master/langid/train/tokenize.py --max_order ${n} multidialect_model_${n}_grams

python langid.py-master/langid/train/DFfeatureselect.py --max_order ${n} multidialect_model_${n}_grams

python langid.py-master/langid/train/IGweight.py -d multidialect_model_${n}_grams

python langid.py-master/langid/train/IGweight.py -lb multidialect_model_${n}_grams

python langid.py-master/langid/train/LDfeatureselect.py multidialect_model_${n}_grams

python langid.py-master/langid/train/scanner.py multidialect_model_${n}_grams

python langid.py-master/langid/train/NBtrain.py multidialect_model_${n}_grams

#python langid.py -m multidialect_model/model