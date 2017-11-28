import email

import charac_feats as cf
import word_feats as wf
import syntac_feats  as syf
import struct_feats as stf
import funct_word_feats as fwf

from nltk.corpus.reader.plaintext import PlaintextCorpusReader

################################################################################

def generate_feat_data(dir_array):
    
    data_list = []
    labels = []

    for direct in dir_array:

        data = []
        
        corpus_dir = 'dataset/' + direct
        corpus = PlaintextCorpusReader(corpus_dir,'.*\.*')
        file_ids = corpus.fileids()
        
        for file in file_ids[:3]:
            
            d = []
            
            raw_email = corpus.raw(file)
            e = email.message_from_string(raw_email)
            
            if(e.is_multipart()):
                for payload in e.get_payload:
                    raw_email = payload.get_payload
            else:
                raw_email = e.get_payload()

            feats = [
                cf.charac_feats_extractor(raw_email),
                wf.word_feats_extractor(raw_email),
                syf.syntac_feats_extractor(raw_email),
                stf.struct_feats_extractor(corpus, file, raw_email),
                fwf.funct_word_feats_extractor(raw_email)
            ]
    
            for f in feats:
                d.extend(list(f.values()))
                
            data.append(d)
            labels.append(1)
            
        data_list.extend(data)
        
    return data_list

male_dirs = ['badeer-r', 'benson-r']
print(generate_feat_data(male_dirs))

    