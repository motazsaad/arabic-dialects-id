
from langid.langid import LanguageIdentifier, model


print('#################################')
print('test 3 -grams model')
arabic_dialect_model3 = open('built_models/exp2/corpus_model_3').read()
identifier3 = LanguageIdentifier.from_modelstring(arabic_dialect_model3, norm_probs=True)

print(identifier3.classify("كيف حالك شو اخبارك انا منيح"))
# ('', )
print(identifier3.classify("ولك من وين انت جاي ياخو ينعن هيك عيشة معبر رفح؟")) #pa
# ('', )
print(identifier3.classify("دخيل قلبك شو مهضوم"))
# ('', )
print(identifier3.classify("ماعاد فينا نتحمل البسينة"))


arabic_dialect_model4 = open('built_models/exp2/corpus_model_4').read()
identifier4 = LanguageIdentifier.from_modelstring(arabic_dialect_model4, norm_probs=True)

print('#################################')
print('test 4-grams model')
print(identifier4.classify("كيف حالك  شو اخبارك انا منيح")) #jo
# ('', )
print(identifier4.classify("ولك من وين انت جاي ياخو ينعن هيك عيشة معبر رفح؟")) #pa
# ('', )
print(identifier4.classify("دخيل قلبك شو مهضوم تؤبر قلبي"))#sy
# ('', )
print(identifier4.classify("ماعاد فينا نتحمل البسينة عم تغط عنفسي"))
# ('', )

print('#################################')
print('test 5-grams model')

arabic_dialect_model5 = open('built_models/exp2/corpus_model_5').read()
identifier5 = LanguageIdentifier.from_modelstring(arabic_dialect_model5, norm_probs=True)

print(identifier5.classify("كيف حالك شو اخبارك انا منيح"))
# ('', )
print(identifier5.classify("ولك من وين انت جاي ياخو ينعن هيك عيشة معبر رفح؟")) #pa
# ('', )
print(identifier5.classify("دخيل قلبك شو مهضوم تؤبر قلبي"))
# ('', )
print(identifier5.classify("ماعاد فينا نتحمل البسينة عم تغط عنفسي"))
# ('', )


print('#################################')
print('test 6 -grams model')
arabic_dialect_model6 = open('built_models/exp2/corpus_model_6').read()
identifier6 = LanguageIdentifier.from_modelstring(arabic_dialect_model6, norm_probs=True)

print(identifier6.classify("كيف حالك شو اخبارك انا منيح"))
# ('', )
print(identifier6.classify("ولك من وين انت جاي ياخو ينعن هيك عيشة معبر رفح؟")) #pa
# ('', )
print(identifier6.classify("دخيل قلبك شو مهضوم تؤبر قلبي"))
# ('', )
print(identifier6.classify("ماعاد فينا نتحمل البسينة عم تغط عنفسي"))


print('#################################')
print('test 7 -grams model')
arabic_dialect_model7 = open('built_models/exp2/corpus_model_7').read()
identifier7 = LanguageIdentifier.from_modelstring(arabic_dialect_model7, norm_probs=True)

print(identifier7.classify("كيف حالك شو اخبارك يا نشمي انا منيح"))
# ('', )
print(identifier7.classify("ولك من وين انت جاي ياخو ينعن هيك عيشة معبر رفح؟")) #pa
# ('', )
print(identifier7.classify("دخيل قلبك شو مهضوم تؤبر قلبي"))
# ('', )
print(identifier7.classify("ماعاد فينا نتحمل البسينة عم تغط عنفسي"))