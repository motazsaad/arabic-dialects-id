#!/usr/bin/env bash


if [ $# -ne 3 ]; then
    echo "usage ${0} python_cmd n train_corpus";
    exit -1;
fi


python_cmd=${1}
n=${2}
train_corpus=${3}



rm -rf multidialect_model_${n}_grams

${python_cmd} langid.py-master/langid/train/index.py ${train_corpus} -m ${train_corpus}_model_${n}_grams

${python_cmd} langid.py-master/langid/train/tokenize.py --max_order ${n} ${train_corpus}_model_${n}_grams

${python_cmd} langid.py-master/langid/train/DFfeatureselect.py --max_order ${n} ${train_corpus}_model_${n}_grams

${python_cmd} langid.py-master/langid/train/IGweight.py -d ${train_corpus}_model_${n}_grams

${python_cmd} langid.py-master/langid/train/IGweight.py -lb ${train_corpus}_model_${n}_grams

${python_cmd} langid.py-master/langid/train/LDfeatureselect.py ${train_corpus}_model_${n}_grams

${python_cmd} langid.py-master/langid/train/scanner.py ${train_corpus}_model_${n}_grams

${python_cmd} langid.py-master/langid/train/NBtrain.py ${train_corpus}_model_${n}_grams

#python langid.py -m multidialect_model/model