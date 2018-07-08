class trainConfig:
    '''
    Set hyperparameters for text classification
    '''
    def __init__(self, max_doc_len=100, label_size=2,
                embed_size=100, hidden_size=250, batch_size=64, layer_size=2)
        self.max_doc_len = max_word_len#Max length of a text
        self.label_size = label_size#The number of different labels
        self.embed_size = embed_size#embedding size of words
        self.hidden_size = hidden_size#hidden size for RNN
        self.batch_size = batch_size#batch size
        self.layer_size = layer_size#layer size