#!/usr/bin/env bash

rm -rf multidialect_model

python langid.py-master/langid/train/index.py train_multidialect_arabic/ -m multidialect_model

python langid.py-master/langid/train/tokenize.py multidialect_model/

python langid.py-master/langid/train/DFfeatureselect.py multidialect_model/

python langid.py-master/langid/train/IGweight.py -d multidialect_model/

python langid.py-master/langid/train/IGweight.py -lb multidialect_model/

python langid.py-master/langid/train/LDfeatureselect.py multidialect_model/

python langid.py-master/langid/train/scanner.py multidialect_model/

python langid.py-master/langid/train/NBtrain.py multidialect_model/

#python langid.py -m multidialect_model/model