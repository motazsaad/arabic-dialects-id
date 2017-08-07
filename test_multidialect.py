from langid.langid import LanguageIdentifier, model
arabic_dialect_model4 = open('multidialect_model/model').read()
identifier4 = LanguageIdentifier.from_modelstring(arabic_dialect_model4, norm_probs=True)

print('#################################')
print('test 4-grams model')
print(identifier4.classify("كيفك شو اخبارك انا منيح"))
# ('jo', 0.9931649962526201)
print(identifier4.classify("انت عملت كده ليه"))
# ('eg', 0.9984723495959299)
print(identifier4.classify("من شان شو"))
# ('pa', 0.6983566048467883)
print(identifier4.classify("صعيبة برشة باهي"))
# ('tn', 0.9997509389390998)

print('#################################')
print('test 5-grams model')

arabic_dialect_model5 = open('multidialect_model_5_grams/model').read()
identifier5 = LanguageIdentifier.from_modelstring(arabic_dialect_model5, norm_probs=True)

print(identifier5.classify("كيفك شو اخبارك انا منيح"))
# ('jo', 0.9990575865088605)
print(identifier5.classify("انت عملت كده ليه"))
# ('eg', 0.9999998593886983)
print(identifier5.classify("من شان شو"))
# ('pa', 0.6756028316826806)
print(identifier5.classify("صعيبة برشة باهي"))
# ('tn', 0.9999999999817093)