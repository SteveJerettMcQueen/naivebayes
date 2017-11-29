from __future__ import division

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import PlaintextCorpusReader

# Total Number of Lines
def num_of_lines(text):
    return 0

# Total Number of Sentences
def num_of_sents(text):
    return len(sent_tokenize(text))

# Total Number of Paragraphs
def num_of_paras(corpus, fid):
    return len(corpus.paras(fid))

# Average Number of Sentences Per Paragraph
def avg_num_of_sents_per_para(corpus, fid, text):
    return num_of_sents(text)/num_of_paras(corpus,fid)

# Average Number of Words Per Paragraph
def avg_num_of_words_per_para(corpus, fid, text):
    num_of_words = len(word_tokenize(text))
    return num_of_words/num_of_paras(corpus, fid)

# Average Number of Characters Per Paragraph
def avg_num_of_chars_per_para(corpus, fid, text):
    return len(text)/num_of_paras(corpus, fid)

# Average Number of Words Per Sentence
def avg_num_of_words_per_sent(text):
    num_of_words = len(word_tokenize(text))
    return (num_of_words/num_of_sents(text))
    
# Number of Sentences Beginning With Upper Case
def num_of_sents_beg_upper_case(text):
    sents = sent_tokenize(text)
    cnt = 0
    
    for s in sents:
        if(s[0].isupper()):
            cnt = cnt + 1
        
    return (cnt/len(sents)) 
    
# Number of Sentences Beginning With Lower Case
def num_of_sents_beg_lower_case(text):
    sents = sent_tokenize(text)
    cnt = 0
    
    for s in sents:
        if(s[0].islower()):
            cnt = cnt + 1

    return (cnt/len(sents)) 

# Number of Blank Lines / Total Number of Lines
def number_of_blank_lines(text):
    return 0
    
# Average Length of Non-Blank Line
def avg_len_of_non_blank_line(text):
    return 0
    
# Feature Extractor
def struct_feats_extractor(corpus, fid, text):
    feats = {}
    
    ns = num_of_sents(text)
    np = num_of_paras(corpus, fid)
    ansp = avg_num_of_sents_per_para(corpus, fid, text)
    anwp = avg_num_of_words_per_para(corpus, fid, text)
    ancp = avg_num_of_chars_per_para(corpus, fid, text)
    anws = avg_num_of_words_per_sent(text)
    nsbu = num_of_sents_beg_upper_case(text)
    nsbl = num_of_sents_beg_lower_case(text)
    
    feats[0] = 1 if (ns > 13) else 0
    feats[1] = 1 if (np < 9.7) else 0
    feats[2] = 1 if (ansp > 1.2) else 0
    feats[3] = 1 if (anwp > 30) else 0
    feats[4] = 1 if (ancp > 185) else 0
    feats[5] = 1 if (anws > 28) else 0
    feats[6] = 1 if (nsbu < .84) else 0
    feats[7] = 1 if (nsbl > .033) else 0

    return feats