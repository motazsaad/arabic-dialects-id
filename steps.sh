#!/usr/bin/env bash


bash build_lang_id_model.sh python2.7 Train_Padic 5
bash build_lang_id_model.sh python Train_Padic 5


bash build_lang_id_model.sh python2.7 Train_Our_Corpus 5
bash build_lang_id_model.sh python Train_Our_Corpus 5
