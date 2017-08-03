from langid.langid import LanguageIdentifier, model
arabic_dialect_model = open('multidialect_model/model').read()
identifier = LanguageIdentifier.from_modelstring(arabic_dialect_model, norm_probs=True)

print(identifier.classify("كيفك شو اخبارك انا منيح"))

print(identifier.classify("انت عملت كده ليه"))

print(identifier.classify("من شان شو"))

print(identifier.classify("صعيبة برشة باهي"))