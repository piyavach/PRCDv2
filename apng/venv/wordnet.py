import nltk
# nltk.download('wordnet')

from nltk.corpus import wordnet as wn
print(wn.synset_from_pos_and_offset('n',1440764))