from langid.langid import LanguageIdentifier, model
arabic_dialect_model = open('multidialect_model/model').read()
identifier = LanguageIdentifier.from_modelstring(arabic_dialect_model, norm_probs=True)

print(identifier.classify("كيفك شو اخبارك انا منيح"))
# ('jo', 0.9931649962526201)
print(identifier.classify("انت عملت كده ليه"))
# ('eg', 0.9984723495959299)
print(identifier.classify("من شان شو"))
# ('pa', 0.6983566048467883)
print(identifier.classify("صعيبة برشة باهي"))
# ('tn', 0.9997509389390998)