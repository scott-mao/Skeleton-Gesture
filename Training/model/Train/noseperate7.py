import tensorflow as tf
from model.op import *
#For segmented gesture recognition training
#Dropout / No layer normalization
class model(object):
    def __init__(self, feature_size, output_size, cell_size,layer,reuse):

        self.feature_size = feature_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.outputall=[]
        self.layer = layer

        self.reuse=reuse
        self.merge = []
        with tf.name_scope('input'):
            self.drop=tf.placeholder(tf.float32,name='Dropout_ratio')
            self.length = tf.placeholder(tf.float32, [None], name='length')  # length of each sequence
            self.lengthmat = tf.placeholder(tf.float32, [None, None, cell_size], name='lengthmat')
            self.input = tf.placeholder(tf.float32, [None, None, feature_size], name='X_in')
            self.ys = tf.placeholder(tf.int64, [None], name='target')  # ys need to be one-hot

        self.buildmodel()
        if self.reuse == True:
                self.merge=tf.summary.merge(self.merge)



    def buildmodel(self):
        if self.reuse ==False:#CPU original model/test model
            self.add_rnn_hidden()
            self.add_output_layer()
            self.accuracy()

        else:
            self.add_rnn_hidden()
            self.add_output_layer()
            self.cross_entropy()
            self.grad()

    def add_rnn_hidden(self):
        with tf.variable_scope('RNN') as scope:
            if self.reuse :
                scope.reuse_variables()

            def gru_cell(d):
                if self.reuse and d:
                    cell = tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=False)
                    return tf.nn.rnn_cell.DropoutWrapper(cell,self.drop)
                else:
                    return tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=False)

            def lstm_cell(d):
                if self.reuse and d:
                    cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, reuse=False)
                    return tf.nn.rnn_cell.DropoutWrapper(cell, self.drop)
                else:
                    return tf.nn.rnn_cell.LSTMCell(self.cell_size, reuse=False)

            output_last=self.input
            for i in range(self.layer):
               with tf.variable_scope('layer'+str(i)):
                   if i > 0:
                      cell=gru_cell(d=True)
                   else:
                      cell=gru_cell(d=False)

                   output_last, state_last = tf.nn.dynamic_rnn(cell
                                                                  , output_last
                                                                  ,dtype=tf.float32
                                                                  ,time_major=True
                                                                  ,sequence_length=self.length
                                                                  ,scope='RNNlayer')
               self.outputall.append(output_last)
            self.output_last= output_last

    def add_output_layer(self):
        with tf.variable_scope('Output') as scope:
            if self.reuse:
                scope.reuse_variables()
           # self.output_last = tf.contrib.layers.layer_norm(self.output_last)

            mod=self.lengthmat*self.output_last
            final_output_seq=tf.reduce_sum(axis=0,input_tensor=mod,name='3_2D')
            final_output_seq = tf.nn.dropout(final_output_seq, self.drop)
            Ws_out = weight_variable([self.cell_size, self.output_size],reuse=self.reuse)
            bs_out = bias_variable([self.output_size, ],reuse=self.reuse)
            with tf.name_scope('Wx_plus_b'):
                temp = tf.matmul(final_output_seq, Ws_out) + bs_out

                max = tf.reduce_max(temp, 1, keep_dims=True)
                temp = temp - max
                self.pred=temp
    def cross_entropy(self):
        with tf.name_scope('Loss') as scope:
            loss= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred,labels=self.ys,name='Cross_entropy')

            self.loss=loss

            self.cost = tf.reduce_mean(loss)
            self.merge.append(tf.summary.scalar('cost', self.cost))

    def grad(self):
        with tf.name_scope('Gradient'):
            tvars=tf.trainable_variables()
            self.grads=tf.gradients(self.cost,tvars)


    def accuracy(self):
        with tf.name_scope('Accuracy') as scope:

             result=tf.nn.softmax(logits=self.pred,name='Predition_labels')
             self.output=result
             self.results=tf.argmax(result, 1)
             correct_prediction = tf.equal(tf.argmax(result, 1), self.ys)
             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='Accuracy')

             tf.summary.scalar('Accuracy', self.accuracy)
             self.merge=tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope))

