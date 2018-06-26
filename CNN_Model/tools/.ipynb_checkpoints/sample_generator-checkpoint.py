import random
import numpy as np
class generate_samples:
    '''Generate samples for training and testing'''
    
    def __init__(self, text_ids, labels, max_len=800):
        '''
        Args:
        text_ids: list of text represented by sequences of ids, 
                    each id can be mapped into a word
        labels: list of labels
        max_len: maximum length for the texts
        '''
        self.index = 0
        self.__text_ids = text_ids
        self.__labels = labels
        self.__max_len = max_len
        self.__text_num = len(text_ids)
        
    def generate_batch(self, batch_size=64, is_training=True):
        '''Generate a training sample each time'''
        
        selected_samples = []
        selected_labels = []
        #For training, select random samples
        if is_training:
            #Record the index of the texts, useful for doc2vec model
            selected_index = np.random.choice(len(self.__text_ids), batch_size, replace=False)
            for index in selected_index:
                selected_samples.append(self.__text_ids[index])
                selected_labels.append(self.__labels[index])
        #For testing, select a few samples each time
        else:#Testing model
            start = self.index%self.__text_num
            end = (start + batch_size)%self.__text_num
            #Record the index of the texts, useful for doc2vec model
            selected_index = np.arange(start, end)
            #In case end goes beyong the range of the samples
            if end > start:
                selected_samples = self.__text_ids[start: end]
                selected_labels = self.__labels[start: end]
                self.index = end
            else:
                print('Test Samples come to an end!')
                selected_samples = self.__text_ids[start: ]
                selected_labels = self.__labels[start: ]
                self.index = 0
            
        #compute the real length for each sentence
        lengths = list(map(len, selected_samples))
        #max_len = self.__max_len
        #Find the max length for the sentences
        max_len = max(lengths)
        #If the specified padding length is shorter than the max length
        max_len = self.__max_len #if max_len > self.__max_len else max_len

        #Create input and label
        #Note, unknow words are mapped to '0'
        x = np.full((batch_size, max_len), 0, np.int32)
        y = np.array(selected_labels)
        lengths = np.array(lengths)
        for i in range(batch_size):
            #the first n elements as input
            if len(selected_samples[i]) < max_len:
                x[i, :len(selected_samples[i])] = selected_samples[i]
                #y[i] = selected_labels[i]
            #If the news is very long
            #Cut it to the max_news_len
            else:
                x[i, :] = selected_samples[i][:max_len]
                #The real length is cut to the max_len
                lengths[i] = max_len
                #y[i] = selected_labels[i]
        return x, y, lengths, selected_index
    