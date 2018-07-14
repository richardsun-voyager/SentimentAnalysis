from collections import Counter
from time import time
from textblob import TextBlob
class token2idx:
    '''
    Build a vacabulary and dictionary of tokens
    Map each token into an unique ID
    Note, a text can be viewed as a series of sentences
    a sentence consist of words
    '''
    def __init__(self, texts, vocab_size=20000):
        '''
        Args:
        texts: a list of words or lists or a list of textblob objects
        vocab_size: length of the vocabulary
        '''
        #sentences
        if isinstance(texts, list):
            self.__texts = texts
        else:
            self.__texts = list(texts)
        self.__vocab_size = vocab_size
        self.__word_idx_dict = None
        self.__idx_word_dict = None
        self.__vocab = None

    def proceed(self, ignore_sent=True):
        '''
        Flatten word lists and build vocabulary
        Args:
        ignore_sent: bool, only valid for textblob object
        '''
        #Flatten word lists
        print('Start mapping words to IDs....')
        start = time()
        words_all = list(self.__flatten__(self.__texts))
        
        #Calculate word frequences
        word_freq_pair = Counter(words_all)
        #Select the most common ones
        if len(word_freq_pair) <= self.__vocab_size:
            #word_common = word_freq_pair
            self.__vocab_size = len(word_freq_pair)
        word_common = word_freq_pair.most_common(self.__vocab_size)
        #Build a vocabulary and dictionaries
        self.__vocab, _ = list(zip(*word_common))
        self.__word_idx_dict = {word: (i+1) for i, word in enumerate(self.__vocab)}
        #Add one more as unknown symbols
        #Set unknow words as '0'
        self.__word_idx_dict['*unknown*'] = 0
        #Build an id-to-word dictionary
        self.__idx_word_dict = dict(zip(self.__word_idx_dict.values(), 
                                         self.__word_idx_dict.keys()))
        #Append unknow to the vocab
        self.__vocab = list(self.__vocab)
        self.__vocab.insert(0, '*unknown')

        #Map the texts into series of IDs
        texts_idx = self.map_text_idx(self.__texts, ignore_sent)
            
        end = time()
        print('Processing Finished! Timing: ', round(end-start, 3))
        return texts_idx

    def map_text_idx(self, texts, ignore_sent):
        '''
        Map text of words into idx, this is for testing data
        '''
        texts_idx = []
        #Traverse each text
        for text in texts:
            text_idx = []
            if (len(text) > 0) and isinstance(text[0], str):
                text_idx = list(map(self.__word2idx__, text))
            elif (len(text) > 0) and isinstance(text[0], list):
                #Traverse each sentence
                for sent in text:
                    #If a text consists of sentences
                    if isinstance(sent, list):
                        sent_idx = list(map(self.__word2idx__, sent))
                        text_idx.append(sent_idx)
            texts_idx.append(text_idx)
        return texts_idx
    
    def get_vocab(self):
        '''
        Return the vocabulary of training data set
        '''
        #vocab = list(self.__vocab)
        #return vocab.insert(0, 'unknown')
        return self.__vocab

    def __word2idx__(self, word):
        '''
        Map a word into an ID
        '''
        try:
            ID = self.__word_idx_dict[word]
        #Unknown words as '0'
        except:
            ID = 0
        return ID

    def __idx2word__(self, ID):
        '''
        Map an ID into a word
        '''
        try:
            word = self.__idx_word_dict[ID]
        except:
            word = '*unknown*'
        return word

    ##https://www.zhihu.com/question/27010691/answer/248832634
    def __flatten__(self, l): 
        '''Flatten all the lists into a big one'''
        for k in l: 
            if not isinstance(k, (list, tuple)): 
                yield k 
            else: 
                yield from self.__flatten__(k)
                
    def __flatten_textblob__(self, blobs):
        '''Flatten all the lists into a big one'''
        words_total = []
        for blob in blobs:
            words_total.extend(blob.words)
        return words_total

