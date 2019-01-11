from langid.langid import LanguageIdentifier, model

arabic_dialect_model4 = open('Padic_model_4_grams/model').read()
identifier4 = LanguageIdentifier.from_modelstring(arabic_dialect_model4, norm_probs=True)

print('#################################')
print('test 4-grams model')
print(identifier4.classify("كيفك شو اخبارك انا منيح"))
# ('', )
print(identifier4.classify("من وين انت جاي؟"))
# ('', )
print(identifier4.classify("دخيل قلبك شو مهضوم"))
# ('', )
print(identifier4.classify("أين تقع المدرسة"))
# ('', )


print('#################################')
print('test 5-grams model')

arabic_dialect_model5 = open('Padic_model_5_grams/model').read()
identifier5 = LanguageIdentifier.from_modelstring(arabic_dialect_model5, norm_probs=True)

print(identifier4.classify("كيفك شو اخبارك انا منيح"))
# ('', )
print(identifier4.classify("من وين انت جاي؟"))
# ('', )
print(identifier4.classify("دخيل قلبك شو مهضوم"))
# ('', )
print(identifier4.classify("أين تقع المدرسة"))
# ('', )



print('#################################')
print('test 6 -grams model')

arabic_dialect_model6 = open('Padic_model_6_grams/model').read()
identifier6 = LanguageIdentifier.from_modelstring(arabic_dialect_model6, norm_probs=True)

print(identifier4.classify("كيفك شو اخبارك انا منيح"))
# ('', )
print(identifier4.classify("من وين انت جاي؟"))
# ('', )
print(identifier4.classify("دخيل قلبك شو مهضوم"))
# ('', )
print(identifier4.classify("أين تقع المدرسة"))
# ('', )


print('#################################')
print('test 7 -grams model')


arabic_dialect_model7 = open('Padic_model_7_grams/model').read()
identifier7 = LanguageIdentifier.from_modelstring(arabic_dialect_model7, norm_probs=True)

print(identifier4.classify("كيفك شو اخبارك انا منيح"))
# ('', )
print(identifier4.classify("من وين انت جاي؟"))
# ('', )
print(identifier4.classify("دخيل قلبك شو مهضوم"))
# ('', )
print(identifier4.classify("أين تقع المدرسة"))
# ('', )
