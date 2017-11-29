import email

import charac_feats as cf
import word_feats as wf
import syntac_feats  as syf
import struct_feats as stf
import funct_word_feats as fwf

from nltk.corpus.reader.plaintext import PlaintextCorpusReader

################################################################################

# Extracts the feature from a text
def extract_features(text, corpus, file):
    return [
        cf.num_of_chars(text),
        cf.num_of_letts(text),
        cf.num_of_upper_chars(text),
        cf.num_of_digit_chars(text),
        cf.num_of_white_space_chars(text),    
        wf.num_of_words(text),
        wf.num_of_short_words(text),
        wf.num_of_words_longer_than_six_chars(text), 
        wf.avg_len_per_word(text), 
        wf.vocab_richness(text), 
        syf.num_of_single_qoutes(text),
        syf.num_of_commas(text),
        syf.num_of_periods(text),
        syf.num_of_colons(text),
        syf.num_of_semi_colons(text),
        syf.num_of_quest_marks(text),
        syf.num_of_excl_marks(text),
        stf.num_of_sents(text),
        stf.num_of_paras(corpus, file),
        stf.avg_num_of_sents_per_para(corpus, file, text),
        stf.avg_num_of_words_per_para(corpus, file, text),
        stf.avg_num_of_chars_per_para(corpus, file, text),
        stf.avg_num_of_words_per_sent(text),
        stf.num_of_sents_beg_upper_case(text),
        stf.num_of_sents_beg_lower_case(text),
        fwf.num_of_pron_words(text),
        fwf.num_of_aux_verbs(text),
        fwf.num_of_conj_words(text),
        fwf.num_of_interj_words(text)
    ]   

# Generates feature set data
def load_feat_data(dir_array):
    
    data_list = []

    for direct in dir_array:
    
        data = []
        
        corpus_dir = 'dataset/' + direct
        corpus = PlaintextCorpusReader(corpus_dir,'.*\.*')
        file_ids = corpus.fileids()
        
        for file in file_ids:
            text = corpus.raw(file)
            e = email.message_from_string(text)
            
            if(e.is_multipart()):
                for payload in e.get_payload:
                    text = payload.get_payload
        
            else:
                text = e.get_payload()
            
            data.append(extract_features(text, corpus, file))
    
        data_list.extend(data)
    
    return data_list

# Loads the data from a set of data and assigned labels
def load_data(dir_label):
    
    data_list = []
    labels = []

    for dl in dir_label:
        
        data = []

        directory = dl[0]
        label = dl[1]
    
        corpus_dir = 'dataset/' + directory
        corpus = PlaintextCorpusReader(corpus_dir,'.*\.*')
        file_ids = corpus.fileids()
        
        for file in file_ids:
            
            d = []
            
            text = corpus.raw(file)
            e = email.message_from_string(text)
            
            if(e.is_multipart()):
                for payload in e.get_payload:
                    text = payload.get_payload
            else:
                text = e.get_payload()

            feats = [
                cf.charac_feats_extractor(text),
                wf.word_feats_extractor(text),
                syf.syntac_feats_extractor(text),
                stf.struct_feats_extractor(corpus, file, text),
                fwf.funct_word_feats_extractor(text)
            ]
    
            for f in feats:
                d.extend(list(f.values()))
                
            data.append(d)
            labels.append(label)
            
        data_list.extend(data)
        
    return [data_list, labels]

# Get column of 2D array    
def column(matrix, i):
    return [row[i] for row in matrix]
