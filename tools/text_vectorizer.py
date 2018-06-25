from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
punctuations = string.punctuation
class text_vectorizer:
    '''
    Clean texts and turn them into vectors
    Args:
    train_data: a list of texts(strings)
    test_data: a list of texts(strings)
    '''
    def __init__(self, train_data, vectorizer, is_lemmatize=True):
        assert isinstance(train_data, list) == True
        self.__train_data = train_data
        self.__vectorizer = vectorizer
        self.__is_lemmatize = is_lemmatize
    
    def proceed(self):
        '''Execute preprocessing and intialize vectorizer'''
        self.__train_data = self.__preprocess(self.__train_data)
        if self.__is_lemmatize:
            self.__train_data = self.__lemmatize(self.__train_data)
        X_train = self.__vectorizer.fit_transform(self.__train_data)
        return X_train
    
    def vectorize(self, texts):
        '''
        Vectorize input texts
        Args:
        texts: list
        '''
        texts = self.__preprocess(texts)
        if self.__is_lemmatize:
            processed_vectors = self.__lemmatize(texts)
        vectors = self.__vectorizer.transform(processed_vectors)
        return vectors

    
    def __preProcessor(self, s):
        '''
        Remove punctuation and digits
        '''
        #remove punctuation
        s = re.sub('['+string.punctuation+']', ' ', s)
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
    
    def __preprocess(self, texts):
        '''Remove punctuations'''
        processed_texts = list(map(self.__preProcessor, texts))
        return processed_texts
    
    def __lemmatize(self, texts):
        '''Lemmatize words into original forms'''
        lemmatizer = WordNetLemmatizer()
        texts_lemmatized = []
        for text in texts:
            lem_data = [lemmatizer.lemmatize(word.strip()) for word in word_tokenize(text)]
            data = ' '.join(lem_data)
            texts_lemmatized.append(data)
        return texts_lemmatized