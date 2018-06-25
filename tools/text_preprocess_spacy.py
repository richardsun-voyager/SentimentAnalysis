from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
from time import time
import spacy
punctuations = string.punctuation
nlp = spacy.load('en')
class text_clean:
    '''
    A class which cleans the original texts by removing punctuation and digits
    '''
    def __init__(self, texts, keep_punctuation=False, is_lemmatize=True):
        assert isinstance(texts, list) == True
        self.__texts = texts
        self.__keep_punctuation = keep_punctuation
        self.__is_lemmatize = is_lemmatize

    def __preprocessor__(self, s):
        '''
        Remove punctuation and digits
        '''
        #remove punctuation
        #s = re.sub('['+string.punctuation+']', ' ', s)
        #remove digits
        s = re.sub('['+string.digits+']', ' ', s)
        #remove foreign characters
        #s = re.sub('[^a-zA-Z]', ' ', s)
        #remove line ends
        s = re.sub('\n', ' ', s)
        #turn to lower case
        s = s.lower()
        s = re.sub('[ ]+',' ', s)
        s = s.rstrip()
        return s
    
    def __preprocess__(self, texts):
        '''Remove punctuations'''
        #processed_texts = [self.__preprocessor__(item) for item in texts]
        processed_texts = list(map(self.__preprocessor__, texts))
        return processed_texts
    
    def __lemmatize_sent__(self, text):
        '''Lemmatize words pf a sentence'''
        #lemmatizer = WordNetLemmatizer()
        #lem_data = [lemmatizer.lemmatize(word.strip()) for word in word_tokenize(text)]
        #lem_sent = ' '.join(lem_data)
        sent = nlp(text)
        lem_data = [token.lemma_ for token in sent]
        lem_sent = ' '.join(lem_data)
        return lem_sent

    def __lemmatize__(self, texts):
        '''Lemmatize words into original forms'''
        texts_lemmatized = list(map(self.__lemmatize_sent__, texts))
        #for text in texts:
            #lem_data = [lemmatizer.lemmatize(word.strip()) for word in word_tokenize(text)]
            #data = ' '.join(lem_data)
            #texts_lemmatized.append(data)
        return texts_lemmatized
    
    def proceed(self):
        '''Execute preprocessing and intialize vectorizer'''
        print('Start to process....')
        start = time()
        try:
            self.__texts = self.__preprocess__(self.__texts)
            if self.__is_lemmatize:
                self.__texts = self.__lemmatize__(self.__texts)
            end = time()
            print('Processing Finished! Timing: ', round(end-start, 3))
            return self.__texts
        except Exception as e:
            print(e)
            return None

    def clean_text(self, text):
        '''
        Clean texts
        '''
        pass