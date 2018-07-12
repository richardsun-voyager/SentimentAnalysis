######Split texts into sentences##############
import re
import nltk
import nltk.data
import string
from time import time
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
import spacy
class text2words:
    '''
    Using spacy tool to tokenize texts into sentences or words
    '''
    def __init__(self, texts):
        '''
        texts: a list of strings, utf-8 encoding
        '''
        self.__nlp__ = spacy.load('en')
        self.__texts__ = texts
        
    def __text2sents__(self, text):
        '''
        Split a text into sentences
        '''
        text = self.__nlp__(text)
        #SPlit into sentences
        sents = list(text.sents)
        sents_words = list(map(self.__sent2words__, sents))
        return sents_words
    
    def __sent2words__(self, sent):
        '''
        Split spacy sentence into spacy tokens
        '''
        sent_words = list(map(self.__to_lower__, sent))
        return sent_words
    
    def __text2words__(self, text):
        '''
        Split a text into words
        '''
        text = self.__nlp__(text)
        words = list(map(self.__to_lower__, text))
        return words
        
    def __to_lower__(self, word):
        '''
        Word: spacy token
        '''
        return word.lemma_.lower()
    
    def proceed(self, is_hierarchical=False):
        '''
        Execute the task
        '''
        tokens = []
        if is_hierarchical:
            tokens = map(self.__text2sents__, self.__texts__)
        else:
            tokens = map(self.__text2words__, self.__texts__)
        return tokens

class text2sents:
    '''
    Split a text into sentences
    '''
    def __init__(self, texts):
        '''
        Args:
        texts: a list of texts
        '''
        assert isinstance(texts, list) == True
        self.__tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.__texts = texts
        
    def __preprocess__(self, text):
        '''
        Remove email address and digits
        '''
        pattern1 = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,5}'
        regex1 = re.compile(pattern1, flags=re.IGNORECASE)
        text = regex1.sub('email', text)
        pattern2 = '[|#$@><=+-]'
        regex2 = re.compile(pattern2)
        text = regex2.sub(' ', text)
        pattern3 = r'\n'
        regex3 = re.compile(pattern3)
        text = regex3.sub(' ', text)
        pattern4 = r'[0-9]'
        regex4 = re.compile(pattern4)
        text = regex4.sub(' ', text)
        pattern5 = r'[ ]+'
        regex5 = re.compile(pattern5)
        text = regex5.sub(' ', text)
        return text.lower()
    
    def __split_text__(self, text):
        #text = self.__preprocess__(text)
        sents = self.__tokenizer.tokenize(text)
        return sents

    def __split_sent__(self, sents):
        s2w = sent2words(sents)
        sent_words = s2w.proceed()
        return sent_words

    def proceed(self, is_hierarchical=True):
        #sents = [self.__split__(text) for text in self.__texts]
        print('Start tokenizing....')
        start = time()
        #Split sentences into words
        if is_hierarchical:
            text_sents = list(map(self.__split_sent__, text_sents))
        else:
            text_sents = list(map(self.__split_text__, self.__texts))
        end = time()
        print('Processing Finished! Timing: ', round(end-start, 3))
        return text_sents

class sent2words:
    '''
    Split a sentence into sequences of words
    '''
    def __init__(self, sents, punctuation=None):
        '''
        Args:
        texts: a list of sentences
        punctuation: a list of symbols that will be removed
        '''
        assert isinstance(sents, list) == True
        self.__tokenizer = WordPunctTokenizer().tokenize
        self.__lemmatizer = WordNetLemmatizer().lemmatize
        self.__sents = sents
        self.__punctuation = punctuation

    def __preprocess__(self, sent):
        '''
        Remove punctuation and digits
        '''
        if self.__punctuation is None:
            return sent
        #remove punctuation
        s = re.sub('['+self.__punctuation+']', ' ', sent)
        #remove digits
        s = re.sub('['+string.digits+']', ' ', s)
        #remove foreign characters
        s = re.sub('[^a-zA-Z]', ' ', s)
        #remove line ends
        s = re.sub('\n', ' ', s)
        #turn to lower case
        s = s.lower()
        s = re.sub('[ ]+',' ', s)
        s = s.rstrip()
        return s
    
    def __split__(self, sent):
        '''
        Split a sentence into words
        Lemmatize words
        '''
        #sent = self.__preprocess__(sent)
        words = self.__tokenizer(sent)    
        words = list(map(self.__lemmatizer, words))
        return words
    
    def proceed(self):
        #self.__texts = self.__preprocess__(self.__texts)
        #print('Start tokenizing....')
        #start = time()
        sent_words = list(map(self.__split__, self.__sents))
        end = time()
        #print('Processing Finished! Timing: ', round(end-start, 3))
        return sent_words
    
    def splitList(self, sents):
        sents_words = list(map(self.__split__, sents))
        return sents_words