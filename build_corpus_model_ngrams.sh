n=4
rm -rf Corpus_model_4_grams_${n}_grams

python langid.py-master/langid/train/index.py train_our_corpus/ -m Corpus_model_4_grams_${n}_grams

#python langid.py-master/langid/train/tokenize.py --max_order ${n} Corpus_model_4_grams_${n}_grams

#python langid.py-master/langid/train/DFfeatureselect.py --max_order ${n} Corpus_model_4_grams_${n}_grams

#python langid.py-master/langid/train/IGweight.py -d Corpus_model_4_grams_${n}_grams

#python langid.py-master/langid/train/IGweight.py -lb Corpus_model_4_grams_${n}_grams

#python langid.py-master/langid/train/LDfeatureselect.py Corpus_model_4_grams_${n}_grams

#python langid.py-master/langid/train/scanner.py Corpus_model_4_grams_${n}_grams

#python langid.py-master/langid/train/NBtrain.py Corpus_model_4_grams_${n}_grams

#python langid.py -m Corpus_model_4_grams/model