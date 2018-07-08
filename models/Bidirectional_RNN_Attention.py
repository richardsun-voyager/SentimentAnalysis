import functools
import tensorflow as tf
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


from tensorflow.contrib import rnn
import numpy as np
class AttRNN_Model:
    def __init__(self, config, is_training=True):
        self.embed_size = config.embed_size
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.label_size = config.label_size
        self.batch_size = config.batch_size
        self.max_doc_len = config.max_doc_len
        self.is_training = is_training
        self.x = tf.placeholder(tf.int32, 
                                [self.batch_size, None])
        self.y = tf.placeholder(tf.int32, [self.batch_size])
        self.lengths = tf.placeholder(tf.int32, [self.batch_size])
        self.predict
        if is_training:
            self.optimize
        print('Model Initialized!')
    
    @lazy_property
    def cost(self):
        logits = self.inference
        targets = tf.one_hot(self.y, self.label_size, 1, 0)
        targets = tf.cast(targets, tf.float32)
        #Note  tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=activation)
        loss = tf.losses.softmax_cross_entropy(targets, logits)
        return loss
    
    @lazy_property
    def predict(self):
        logits = self.inference
        #probs = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)
        return predictions
    
    @lazy_property
    def correct_num(self):
        prediction = self.predict
        targets = tf.reshape(self.y, [-1])
        targets = tf.cast(targets, tf.int64)
        correct_prediction = tf.equal(prediction, targets)
        correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        return correct_num
    
    @lazy_property
    def optimize(self):
        with tf.variable_scope('optimizer'):
            cost = self.cost
        #with tf.name_scope('Optimizer'):
            #self._learning_rate = tf.Variable(0.0, trainable=False)
            train_op = tf.train.AdamOptimizer(0.0001).minimize(cost)
            #train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(cost)
            #tvars = tf.trainable_variables()
            #grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 6)
            #optimizer = tf.train.AdamOptimizer(self._learning_rate)
            #train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op
    
    @lazy_property
    def inference(self):
        #Create embedding matrix
        with tf.device("/cpu:0"):
            embeddings = tf.get_variable('embedding', [self.vocab_size,  self.embed_size])
            inputs = tf.nn.embedding_lookup(embeddings, self.x)
        if self.is_training:
            inputs = tf.nn.dropout(inputs, 0.5)

        def lstm():
            return rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, 
                                      state_is_tuple=True) 
        
        def GRU():
            return rnn.GRUCell(self.hidden_size)
        #lstm_cell = lstm
        #cell = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], 
                                #state_is_tuple=True)
        fw_cell = GRU()
        bw_cell = GRU()
        initial_fw_state = fw_cell.zero_state(self.batch_size, tf.float32)
        initial_bw_state = bw_cell.zero_state(self.batch_size, tf.float32)
        #Bidirectional dynamic RNN with given lengths for each text
        outputs, status = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                        initial_state_fw=initial_fw_state,
                                                        initial_state_bw=initial_bw_state,
                                                          sequence_length=self.lengths,
                                                          dtype=tf.float32)
        #Use bidirectional rnn output as hidden states for words
        #size=batch_size, word_length, word_embedding_size*2
        #H = tf.concat([outputs[0], outputs[1], inputs], axis=2)
        H = tf.concat([outputs[0], outputs[1], inputs], axis=2)

        
        #Calculate attention weights for each word
        with tf.variable_scope('Self_Attention'):
            #W_att = tf.get_variable('word_attention_weights', [self.hidden_size*2, 64])
            #b_att = tf.get_variable('word_attention_biases', [64])
            W_u = tf.get_variable('attention_softmax_weights', [64, 1])
            S = []
            for i in np.arange(self.batch_size):
                #Calculate the coefficients of attention
                h = H[i, :, :]
                u = tf.layers.dense(h, 64, activation=tf.tanh)
                #u = tf.tanh(tf.matmul(h, W_att) + b_att)
                #Softmax
                A = tf.nn.softmax(tf.matmul(u, W_u))
                #Transform original representation into a sum of weighted hidden states
                s = tf.reduce_sum(A * h, 0)
                S.append(s)
        
        #Put all the elements within the list into a tensor
        S = tf.stack(S)
        
        #Output layer   
        with tf.variable_scope('output_layer'):
            logits = tf.layers.dense(S, self.label_size, activation=None)
        #预测值
        return logits
    
    @property
    def learningRate(self):
        return self._learning_rate