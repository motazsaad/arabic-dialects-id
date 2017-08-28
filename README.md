# arabic-dialects-id
Arabic dialects identification system 

Steps to prepare a dataset for training 
1. each dialect in one file 
2. split each file into train (90) and test (10)
```split -l $[ $(wc -l filename|cut -d" " -f1) * 90 / 100 ] filename```
3. split each train into lines 
```split -l 1 -a 4 -d file.ext prefix ara_``` 
4. prepare directory structure:
train_corpus/domain/lang/docs/

note : corpus_model_n_grams    old model builded on old version of data without preorocessing

