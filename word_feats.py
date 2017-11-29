from __future__ import division
from collections import Counter

from nltk.tokenize import word_tokenize

# Total Number of Words
def num_of_words(text):
    return len(word_tokenize(text))
    
# Average length per word in characters
def avg_len_per_word(text):
    num_of_chars = len(text)
    words = word_tokenize(text)
    word_char_length = 0
    
    for w in words:
        word_char_length += len(w)
    
    return (word_char_length/num_of_chars) 

# Vocabulary Richness
def vocab_richness(text):
    words = word_tokenize(text)
    uniq_words = set(word_tokenize(text))
    return (len(uniq_words)/len(words)) 
    
# Words longer than 6 characters
def num_of_words_longer_than_six_chars(text):
    words = word_tokenize(text)
    uniq_words = set(words)
    cnt = 0
    for w in uniq_words:
        if (len(w) > 6):
            cnt += 1
    
    return (cnt/len(words)) 

# Words with lengths between [1 -3] characters
def num_of_short_words(text):
    words = word_tokenize(text)
    uniq_words = set(words)
    cnt = 0
    for w in uniq_words:
        if (0 < len(w) < 4):
            cnt += 1
            
    return (cnt/len(words)) 

# Number of net abbreviations
def num_of_net_abbr(text):
    return 0

# Feature Extractor
def word_feats_extractor(text):
    feats = {}

    nw = num_of_words(text)
    nsw = num_of_short_words(text)
    nwlsc = num_of_words_longer_than_six_chars(text)
    alpw = avg_len_per_word(text), 
    vr = vocab_richness(text)
    
    feats[0] = 1 if (nw > 310) else 0
    feats[1] = 1 if (nsw > .215) else 0
    feats[2] = 1 if (nwlsc < .19) else 0
    feats[3] = 1 if (alpw > .822) else 0
    feats[4] = 1 if (vr > .628) else 0

    return feats