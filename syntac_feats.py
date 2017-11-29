from __future__ import division
from collections import Counter

# Total number of single qoutes 
def num_of_single_qoutes(text):
    c = Counter(text)
    return (c["'"]/len(text)) 
    
# Total number of commas 
def num_of_commas(text):
    c = Counter(text)
    return c[","]/len(text) 
    
# Total number of periods 
def num_of_periods(text):
    c = Counter(text)
    return c["."]/len(text) 
    
# Total number of colons 
def num_of_colons(text):
    c = Counter(text)
    return c[":"]/len(text) 
    
# Total number of semi colons 
def num_of_semi_colons(text):
    c = Counter(text)
    return c[";"]/len(text) 
    
# Total number of question marks 
def num_of_quest_marks(text):
    c = Counter(text)
    return c["?"]/len(text) 
    
# Total number of multiple question marks 
def num_of_mult_quest_marks(text):
    c = Counter(text)
    return 0

# Total number of exclamation marks 
def num_of_excl_marks(text):
    c = Counter(text)
    return c["!"]/len(text) 
  
# Total number of multiple exclamation marks 
def num_of_mult_excl_marks(text):
    c = Counter(text)
    return 0
      
# Total number of ellipsis 
def num_of_ellips(text):
    return 0
    
# Feature Extractor
def syntac_feats_extractor(text):
    feats = {}
    
    nsq = num_of_single_qoutes(text)
    nc = num_of_commas(text)
    np = num_of_periods(text)
    ncln = num_of_colons(text)
    nscln = num_of_semi_colons(text)
    nqm = num_of_quest_marks(text)
    nem = num_of_excl_marks(text)
    
    feats[0] = 1 if (nsq > .0013) else 0
    feats[1] = 1 if (nc < .0080) else 0
    feats[2] = 1 if (np > .011) else 0
    feats[3] = 1 if (ncln < .012) else 0
    feats[4] = 1 if (nscln < .0017) else 0
    feats[5] = 1 if (nqm > .0009) else 0
    feats[6] = 1 if (nem > .00043) else 0
    
    return feats
