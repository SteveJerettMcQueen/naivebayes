from __future__ import division
from collections import Counter

import re

from nltk.tokenize import word_tokenize

# Total number of Characters in words
def num_of_chars(text):
    return len(text)

# Total number of Letters
def num_of_letts(text):
    c = Counter(text.lower())
    p = re.compile('[a-z]+')
    cnt = 0;
    for k,v in c.items():
        if(p.match(k)):
            cnt += v

    return (cnt/num_of_chars(text)) 

# Total number of Upper case characters for each word
def num_of_upper_chars(text):
    words = word_tokenize(text)
    c = Counter(words)
    p = re.compile('[A-Z]+')
    cnt = 0;
    for k,v in c.items():
        if(p.match(k)):
            cnt += v

    return (cnt/num_of_chars(text)) 

# Total number of digital characters 
def num_of_digit_chars(text):
    c = Counter(text)
    p = re.compile('[0-9]+')
    cnt = 0;
    for k,v in c.items():
        if(p.match(k)):
            # print(k,v)
            cnt += v
            
    return (cnt/num_of_chars(text)) 

# Total number of white space characters
def num_of_white_space_chars(text):
    c = Counter(text)
    p = re.compile(' ')
    cnt = 0;
    for k,v in c.items():
        if(p.match(k)):
            cnt += v
    
    return (cnt/num_of_chars(text)) 

# Total number of Tab space characters
def num_of_tab_space_chars(text):
    # c = Counter(text)
    # p = re.compile("\t")
    # cnt = 0;
    # for k,v in c.items():
    #     if(p.match(k)):
    #         print(k,v)

    # return (cnt/num_of_chars(text)) * 100
    return 0
    
# Total number of special characters 
def num_of_spec_chars(text):
    # c = Counter(text)
    # p = re.compile('[~`!@#$%^&*()_-+={}[]:>;,</?*-+]')
    # cnt = 0;
    # for k,v in c.items():
    #     if(p.match(k)):
    #         print(k,v)
    # #         cnt += v
    
    # # return (cnt/num_of_chars(text)) * 100
    return 0

# Feature extractor
def charac_feats_extractor(text):
    feats = {}
    
    nc = num_of_chars(text)
    nl = num_of_letts(text)
    nup = num_of_upper_chars(text)
    ndc = num_of_digit_chars(text)
    nwsc = num_of_white_space_chars(text)
    
    feats[0] = 1 if (nc < 1630) else 0
    feats[1] = 1 if (nl < .71) else 0
    feats[2] = 1 if (nup < .046) else 0
    feats[3] = 1 if (ndc < .038) else 0
    feats[4] = 1 if (nwsc > .14) else 0

    return feats