
from langid.langid import LanguageIdentifier, model

import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--number_of_grams", "-n", type=str, help='enter number of grams', required=True)

args = parser.parse_args()
docs = ["ولك من وين انت جاي ياخو ينعن هيك عيشة معبر رفح؟","كيف حالك  شو اخبارك انا منيح","دخيل قلبك شو مهضوم تؤبر قلبي","ماعاد فينا نتحمل البسينة عم تغط عنفسي"]



arabic_dialect_model = open('Train_Pure_corpus_model_'+args.number_of_grams+'_grams/model').read()#open('built_models/exp2/corpus_model_4').read()
identifier = LanguageIdentifier.from_modelstring(arabic_dialect_model, norm_probs=True)

print('#################################')
print('test '+args.number_of_grams+'-grams model')
for doc in docs :
    print(doc, identifier.classify(doc)) #jo


