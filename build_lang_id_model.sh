#!/usr/bin/env bash


if [ $# -ne 3 ]; then
    echo "usage ${0} python_cmd train_corpus n";
    exit -1;
fi


python_cmd=${1}
train_corpus=${2}
n=${3}


printf "python_cmd: %s\n" "${python_cmd}"
printf "n gram: %s\n" "${n}"
printf "train_corpus: %s\n" "${train_corpus}"

rm -rf ${train_corpus}_model_${n}_grams


echo "step 1: make index"
${python_cmd} langid.py-master/langid/train/index.py ${train_corpus} -m ${train_corpus}_model_${n}_grams


echo "step 2: tokenization"
${python_cmd} langid.py-master/langid/train/tokenize.py --max_order ${n} ${train_corpus}_model_${n}_grams


echo "step 3: feature selection"
${python_cmd} langid.py-master/langid/train/DFfeatureselect.py --max_order ${n} ${train_corpus}_model_${n}_grams

echo "step 4: IG weight"
${python_cmd} langid.py-master/langid/train/IGweight.py -d ${train_corpus}_model_${n}_grams

${python_cmd} langid.py-master/langid/train/IGweight.py -lb ${train_corpus}_model_${n}_grams

echo "step 5: LD feature selection"
${python_cmd} langid.py-master/langid/train/LDfeatureselect.py ${train_corpus}_model_${n}_grams


echo "step 6: train scanner"
${python_cmd} langid.py-master/langid/train/scanner.py ${train_corpus}_model_${n}_grams

echo "step 7: NB training"
${python_cmd} langid.py-master/langid/train/NBtrain.py ${train_corpus}_model_${n}_grams

#python langid.py -m multidialect_model/model