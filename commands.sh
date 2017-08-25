#!/usr/bin/env bash



# langid training need python 2.7

# uncomment/comment the suitable python version for you
#py=python2.7
py=python # some machines python means version 2.7



# Padic corpus
bash build_lang_id_model.sh ${py} Train_Padic 4
bash build_lang_id_model.sh ${py} Train_Padic 5







bash build_lang_id_model.sh python2.7 Train_Padic 5
bash build_lang_id_model.sh python Train_Padic 5


bash build_lang_id_model.sh python2.7 Train_Our_Corpus 5
bash build_lang_id_model.sh python Train_Our_Corpus 5


bash build_lang_id_model.sh python2.7 train_multidialect_arabic 7