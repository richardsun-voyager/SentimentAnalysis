#from gensim import utils
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
path = './tools'
sys.path.append(path)


def read_data():
    train_data = pd.read_csv('./data/IMDB_review_train.csv')
    test_data = pd.read_csv('./data/IMDB_review_test.csv')
    train_texts = list(train_data.text.values)
    train_labels = list(train_data.sentiment.values)
    test_texts = list(test_data.text.values)
    test_labels = list(test_data.sentiment.values)
    return train_texts, test_texts, train_labels, test_labels

if __name__ == '__main__':
    train_texts, test_texts, train_labels, test_labels = read_data()
    from text_preprocess import text_clean
    tc = text_clean(train_texts)
    train_processed = tc.proceed()
    tc = text_clean(test_texts)
    test_processed = tc.proceed()
    from text_hier_split import sent2words
    from token_idx_map import token2idx
    sw_train = sent2words(train_texts)
    sw_test = sent2words(test_texts)
    train_sent_words = sw_train.proceed()
    test_sent_words = sw_test.proceed()
    ti = token2idx(train_sent_words, 30000)
    train_sent_idx = ti.proceed()
    test_sent_idx = ti.map_text_idx(test_sent_words, ignore_sent=True)
    n_words = len(ti.get_vocab())
    print('Total words: %d' % n_words)

    from sample_generator import generate_samples
    MAX_DOCUMENT_LENGTH = 800
    gs_train = generate_samples(train_sent_idx, train_labels, MAX_DOCUMENT_LENGTH)
    gs_test = generate_samples(test_sent_idx, test_labels, MAX_DOCUMENT_LENGTH)

    class trainConfig:
        vocab_size = n_words
        max_doc_len = MAX_DOCUMENT_LENGTH
        label_size = 2
        embed_size = 64
        hidden_size = 64
        batch_size = 64
        layer_size = 2
    
    class testConfig:
        vocab_size = n_words
        max_doc_len = MAX_DOCUMENT_LENGTH
        label_size = 2
        embed_size = 64
        hidden_size = 64
        batch_size = 64
        layer_size = 2
    
    class singleConfig:
        vocab_size = n_words
        max_doc_len = MAX_DOCUMENT_LENGTH
        label_size = 2
        embed_size = 64
        hidden_size = 64#hidden size for hidden state of rnn
        batch_size = 1

    train_chunk_num = int(len(train_texts)/trainConfig.batch_size)
    test_chunk_num = int(len(test_texts)/trainConfig.batch_size)
    remain_num = len(test_texts) - trainConfig.batch_size*test_chunk_num
    print(remain_num)

    


