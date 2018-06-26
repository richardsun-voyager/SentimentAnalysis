#from gensim import utils
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import tensorflow as tf
path = './tools'
sys.path.append(path)


def read_data():
    '''
    Read data from local file
    '''
    train_data = pd.read_csv('./data/IMDB_review_train.csv')
    test_data = pd.read_csv('./data/IMDB_review_test.csv')
    train_texts = list(train_data.text.values)
    train_labels = list(train_data.sentiment.values)
    test_texts = list(test_data.text.values)
    test_labels = list(test_data.sentiment.values)
    return train_texts, test_texts, train_labels, test_labels

def process_data(train_texts, test_texts):
    '''
    Clean the texts and map words into IDs
    '''
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
    return train_sent_idx, test_sent_idx, ti


if __name__ == '__main__':
    train_texts, test_texts, train_labels, test_labels = read_data()
    train_sent_idx, test_sent_idx, ti = process_data(train_texts, test_texts)
    n_words = len(ti.get_vocab())
    print('Total words: %d' % n_words)
    print('Data Processed!')

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

    print('Build Graph')
    from model import CNN_Model
    graph_cnn = tf.Graph()
    #Create models for training and testing data
    with graph_cnn.as_default():
        initializer = tf.random_uniform_initializer(-0.02, 0.02)
        with tf.name_scope('train'):
            train_data = tf.placeholder(tf.int32, [trainConfig.batch_size, MAX_DOCUMENT_LENGTH])
            train_label = tf.placeholder(tf.int32, [trainConfig.batch_size])
            train_lengths = tf.placeholder(tf.int32, [trainConfig.batch_size])
            #Set different models for different buckets
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_model = CNN_Model(trainConfig, train_data, train_label, train_lengths)
                saver=tf.train.Saver()
        with tf.name_scope('test'):
            test_data = tf.placeholder(tf.int32, [testConfig.batch_size, None])
            test_label = tf.placeholder(tf.int32, [testConfig.batch_size])
            test_lengths = tf.placeholder(tf.int32, [testConfig.batch_size])
            single_data = tf.placeholder(tf.int32, [singleConfig.batch_size, None])
            single_label = tf.placeholder(tf.int32, [singleConfig.batch_size])
            single_lengths = tf.placeholder(tf.int32, [singleConfig.batch_size])
            #Set different models for different buckets
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = CNN_Model(testConfig, test_data, test_label, test_lengths, False)
                single_model = CNN_Model(singleConfig, single_data, single_label, single_lengths, False)

    
    import time, os
    epochs = 3
    #train_chunk_num = 10
    file = "ckpt_cnn/cnn.ckpt"
    with tf.Session(graph=graph_cnn) as sess:
        #Initialize parameters
        init = tf.global_variables_initializer()
    
        if os.path.exists("ckpt_cnn/cnn.ckpt.index"):
            saver.restore(sess, file)
        else:
            sess.run(init)
        start_time = time.time()
        for m in range(epochs):
            for i in range(train_chunk_num):
                #sess.run(tf.assign(learning_rate, 0.002*((0.98)**m)))
                x, y, lengths, _ = gs_train.generate_batch(trainConfig.batch_size)
                feed_dict = {train_data:x, train_label:y, train_lengths:lengths}
                l, _ = sess.run([train_model.cost, train_model.optimize], feed_dict=feed_dict)
                if i%100 == 0:
                    print('Loss:', round(l, 4))
            end_time = time.time()
            print('Epoch', m, 'time:{:.2f}'.format(end_time - start_time))
            start_time = end_time
        saver.save(sess,'ckpt_cnn/cnn.ckpt')
        #Calculate Testing Accuracy
        print('Testing...')
        count = 0
        gs_test = generate_samples(test_sent_idx, test_labels, MAX_DOCUMENT_LENGTH)
        for _ in range(test_chunk_num):
            #Traverse each data
            x, y, lengths, _ = gs_test.generate_batch(testConfig.batch_size, False)
            feed_dict = {test_data:x, test_label:y, test_lengths:lengths}
            n = sess.run(test_model.correct_num, feed_dict=feed_dict)
            count += np.sum(n)
        for _ in range(remain_num):
            #Traverse each data
            x, y, lengths, _ = gs_test.generate_batch(1, False)
            feed_dict = {single_data:x, single_label:y, single_lengths:lengths}
            n = sess.run(single_model.correct_num, feed_dict=feed_dict)
            count += np.sum(n)
        end_time = time.time()
        print('Testing Time:{:.2f}'.format(end_time - start_time))
        print(count*1.0/len(test_texts))  

    


