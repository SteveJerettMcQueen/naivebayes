from __future__ import division

import nltk

from collections import Counter
from nltk.tokenize import word_tokenize

# Number of Article Words
def num_of_artcl_words(text):
    return 0
    
# Number of Pronoun Words
def num_of_pron_words(text):
    words = word_tokenize(text)
    return get_pos_num(
        'PRP', words, 
        nltk.pos_tag(words),
    )

# Number of Auxiliary Verbs
def num_of_aux_verbs(text):
    words = word_tokenize(text)
    return get_pos_num(
        'MD', words, 
        nltk.pos_tag(words),
    )
    
# Number of Conjuction Words
def num_of_conj_words(text):
    words = word_tokenize(text)
    return get_pos_num(
        'JJ', words, 
        nltk.pos_tag(words),
    )
    
# Number of Interjection Words
def num_of_interj_words(text):
    words = word_tokenize(text)
    return get_pos_num(
        'UH', words, 
        nltk.pos_tag(words),
    )

# Number of Gender-Specific Words
def num_of_gender_specf_words(text):
    return 0

# Helper function
def get_pos_num(pos_tag, words, pos_pairs):
    cnt = 0
    
    for p in pos_pairs:
        c = Counter(p)
        
        if (c[pos_tag] == 1):
            cnt += 1

    return (cnt/len(words))
    
# Feature Extractor
def funct_word_feats_extractor(text):
    feats = {}
    
    npw = num_of_pron_words(text)
    naw = num_of_aux_verbs(text)
    ncj = num_of_conj_words(text)
    nij = num_of_interj_words(text)

    feats[0] = 1 if (npw < .031) else 0
    feats[1] = 1 if (naw > .0115) else 0 
    feats[2] = 1 if (ncj > .041) else 0
    feats[3] = 1 if (nij > .00012) else 0

    return feats
    
    