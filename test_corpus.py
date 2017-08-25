from langid.langid import LanguageIdentifier, model
arabic_dialect_model4 = open('Train_Our_Corpus_model_4_grams/model').read()
identifier4 = LanguageIdentifier.from_modelstring(arabic_dialect_model4, norm_probs=True)

print('#################################')
print('test 4-grams model')
print(identifier4.classify("كيفك شو اخبارك انا منيح"))
# ('jo', )
print(identifier4.classify("انت عملت كده ليه"))
# ('eg', )
print(identifier4.classify("من شان شو"))
# ('pa', )
print(identifier4.classify("صعيبة برشة باهي"))
# ('tn', )

print('#################################')
print('test 5-grams model')

arabic_dialect_model5 = open('Corpus_model_5_grams/model').read()
identifier5 = LanguageIdentifier.from_modelstring(arabic_dialect_model5, norm_probs=True)

print(identifier5.classify("كيفك شو اخبارك انا منيح"))
# ('jo', )
print(identifier5.classify("انت عملت كده ليه"))
# ('eg', )
print(identifier5.classify("من شان شو"))
# ('pa', )
print(identifier5.classify("صعيبة برشة باهي"))
# ('tn', )

print('#################################')
print('test 6 -grams model')

arabic_dialect_model6 = open('Corpus_model_6_grams/model').read()
identifier6 = LanguageIdentifier.from_modelstring(arabic_dialect_model6, norm_probs=True)

print(identifier5.classify("كيفك شو اخبارك انا منيح"))
# ('jo', )
print(identifier5.classify("انت عملت كده ليه"))
# ('eg', )
print(identifier5.classify("من شان شو"))
# ('pa', )
print(identifier5.classify("صعيبة برشة باهي"))
# ('tn', )

print('#################################')
print('test 7 -grams model')


arabic_dialect_model7 = open('Corpus_model_7_grams/model').read()
identifier7 = LanguageIdentifier.from_modelstring(arabic_dialect_model7, norm_probs=True)

print(identifier5.classify("كيفك شو اخبارك انا منيح"))
# ('jo', )
print(identifier5.classify("انت عملت كده ليه"))
# ('eg', )
print(identifier5.classify("من شان شو"))
# ('pa', )
print(identifier5.classify("صعيبة برشة باهي"))
# ('tn', )