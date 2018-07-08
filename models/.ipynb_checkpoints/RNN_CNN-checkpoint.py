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
class RNN_CNN_Model:
    '''
    A CNN followed by a RNN
    CNN can capture n-gram information
    RNN can remember dependency
    '''
    def __init__(self, config, is_training=True):
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.label_size = config.label_size
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.max_doc_len = config.max_doc_len
        self.is_training = is_training
        self.x = tf.placeholder(tf.int32, 
                                [self.batch_size, config.max_doc_len])
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
            #Batch_size, word_length, embed_size
            inputs = tf.nn.dropout(inputs, 0.5)
            
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
        #In a dynamic RNN, if the length is N, the outputs for the words after N are 0
        #And the status are copycats of the last status of the Nth word
        #Use bidirectional rnn output as hidden states for words
        #Outputs, batch_size, max_len, hidden_size*2
        output = tf.concat([outputs[0], outputs[1]], axis=2)
        output_expand = tf.expand_dims(output, 3)
        #print(output)
        #print(output_expand)
        with tf.variable_scope("conv-maxpool"):
            # Convolution Layer
            #batch_size, max_len-1, 1, 256
            h = tf.layers.conv2d(output_expand, 256,
                                       kernel_size=(2, 2*self.hidden_size),
                                       padding='valid',
                                 activation=tf.nn.relu)
                
    
            h_max_pool = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_doc_len-1, 
                           1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            h_avg_pool = tf.nn.avg_pool(
                    h,
                    ksize=[1, self.max_doc_len-1, 
                           1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
        
        #h_max_pool_squeeze = tf.squeeze(h_max_pool)
        #h_avg_pool_squeeze = tf.squeeze(h_avg_pool)
        h_pool = tf.concat([h_max_pool, h_avg_pool], axis=3)
        h_pool = tf.squeeze(h_pool, [1, 2])
        
        h_pool_flat = tf.reshape(h_pool, [self.batch_size, -1])
        
        if self.is_training:
            h_pool_flat = tf.nn.dropout(h_pool_flat, 0.5)
        
        
        with tf.variable_scope('output'):
            logits = tf.layers.dense(h_pool_flat, self.label_size, 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        
        return logits
    
    @property
    def learningRate(self):
        return self._learning_rate
        
        