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
class CNN_RNN_Model:
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
            #Batch_size, word_length, embed_size
            inputs = tf.nn.dropout(inputs, 0.5)
            
        def GRU():
            return rnn.GRUCell(self.hidden_size)
        #Mask those padding words as 0, others as 11
        mask = tf.sequence_mask(self.lengths, self.max_doc_len)
        mask = tf.cast(mask, dtype=tf.float32)
        #Duplicate binary values for the embeddings
        #Those embeddings for the padding words are masked
        #Broadcast embed_size times
        mask_transient = tf.tile(mask, [1, self.embed_size])
        mask_embed = tf.transpose(tf.reshape(mask_transient, [self.batch_size, 
                                                              self.embed_size,
                                                self.max_doc_len]), [0, 2, 1])
        mask_inputs = tf.multiply(mask_embed, inputs)
        #Expand the dim to cater to CNN
        #Batch_size, word_length, embed_size, 1
        intputs_expanded = tf.expand_dims(mask_inputs, -1)
        #print(intputs_expanded)
        
        #Three kinds of convolutional kernels, with kernel size 2, 3, 4
        filter_sizes = [2, 3, 4]
        num_filters = 256
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                h = tf.layers.conv2d(intputs_expanded, num_filters,
                                       kernel_size=(filter_size, self.embed_size),
                                       strides=(1, 1), padding='valid',
                                        activation=tf.nn.relu)
                
    
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, filter_size, 
                           1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                #GRU
                pooled = tf.squeeze(pooled, 2)
                cell = GRU()
                lengths = self.lengths + 2 - 2*filter_size
                initial_state = cell.zero_state(self.batch_size, tf.float32)
                outputs, status = tf.nn.dynamic_rnn(cell, pooled,
                                                          initial_state=initial_state,
                                                          sequence_length=lengths,
                                                          dtype=tf.float32)
                #print(status)
                pooled_outputs.append(status)
                #print(pooled)
            #self.weights.append(W)
            
        
        #num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 1)
        #print(h_pool)
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
        