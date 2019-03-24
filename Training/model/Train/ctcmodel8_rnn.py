#Seperate finger information at input layer and the first RNN layer
#CTC model
import tensorflow as tf
from model.op import *

class model(object):
    def __init__(self, feature_size, output_size, cell_size,layer,part,reuse):

        self.feature_size = feature_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.outputall=[]
        self.layer = layer-1
        self.merge=[]
        self.reuse=reuse
        self.part = part
        with tf.name_scope('input'):
            self.drop=tf.placeholder(tf.float32,name='Dropout_ratio')
            self.length = tf.placeholder(tf.int32, [None], name='length')  # length of each sequence
            self.input = tf.placeholder(tf.float32, [None, None, feature_size], name='X_in')
            self.ys = tf.sparse_placeholder(tf.int32, name='target')  # ys need to be one-hot

        self.buildmodel()
        if self.reuse == False:

            self.merge=tf.summary.merge(self.merge)



    def buildmodel(self):
        if self.reuse ==False:#CPU original model/test model
            #self.add_input_layer()
            self.add_input()
            self.add_rnn_hidden()
            self.add_output_layer()
            self.ctc_loss()
            self.optimize()

        else:
            #self.add_input_layer()
            self.add_input()
            self.add_rnn_hidden()
            self.add_output_layer()
            self.accuracy()
            self.ctc_loss()
            self.costsum=tf.summary.scalar('tcost',self.cost)
            #self.grad()

    def add_input(self):
        with tf.variable_scope('NN') as scope:
            if self.reuse:
                scope.reuse_variables()
            tempout=[]
            input_part=[self.input[:,:,0:3],
                        self.input[:,:,3:6],
                        self.input[:,:,   6:18],
                        self.input[:, :, 18:30],
                        self.input[:, :, 30:42],
                        self.input[:, :, 42:54],
                        self.input[:, :, 54:66]]#W C T I M R P

            for ind,part in enumerate(input_part):
                with tf.variable_scope('part'+str(ind)) as scope2:
                    if self.reuse:
                        scope2.reuse_variables()

                    shape=part.get_shape()
                    inshape=shape[2].value
                    outshape=self.part[ind]
                    cell=tf.nn.rnn_cell.GRUCell(outshape, reuse=False)
                    #cell=tf.nn.rnn_cell.BasicRNNCell(outshape,reuse=False)
                    output_last, state_last = tf.nn.dynamic_rnn(cell
                                                                , part
                                                                , dtype=tf.float32
                                                                , time_major=True
                                                                , sequence_length=self.length
                                                                , scope='RNNlayer')
                    output_last=tf.contrib.layers.layer_norm(output_last)
                    tempout.append( output_last)

            self.inputcombine=tf.concat(tempout,2)
           # self.inputcombine=tf.contrib.layers.layer_norm(self.inputcombine)


    def add_rnn_hidden(self):
        with tf.variable_scope('RNN') as scope:
            if self.reuse :
                scope.reuse_variables()

            def gru_cell():
                    cell=tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=False)
                    return cell
            def lstm_cell():
                return tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, reuse=False)

                #return tf.contrib.rnn.LayerNormBasicLSTMCell(self.cell_size, reuse=False)

            output_last=self.inputcombine
            for i in range(self.layer):
               with tf.variable_scope('layer'+str(i)):
                if i>0:
                    output_last = tf.contrib.layers.layer_norm(output_last)

                output_last, state_last = tf.nn.dynamic_rnn(gru_cell()
                                                                  , output_last
                                                                  ,dtype=tf.float32
                                                                  ,time_major=True
                                                                  ,sequence_length=self.length
                                                                  ,scope='RNNlayer')
                self.outputall.append(output_last)
                #output_last=tf.nn.dropout(output_last,self.drop)
            self.output_last= output_last

    def add_output_layer(self):
        with tf.variable_scope('Output') as scope:
            if self.reuse:
                scope.reuse_variables()

            self.output_last = tf.contrib.layers.layer_norm(self.output_last)
            #self.output_last=tf.nn.dropout(self.output_last,self.drop)
            outall = tf.reshape(self.output_last, [-1, self.cell_size])

            Ws_out = weight_variable([self.cell_size, self.output_size],reuse=self.reuse)
            bs_out = bias_variable([self.output_size, ],reuse=self.reuse)
            with tf.name_scope('Wx_plus_b'):
                temp = tf.matmul(outall, Ws_out) + bs_out
                max=tf.reduce_max(temp,1,keep_dims=True)
                temp=temp-max
                shape = tf.shape(self.output_last)
                self.pred = tf.reshape(temp, [shape[0], shape[1], self.output_size])

    def ctc_loss(self):
        with tf.name_scope('Loss') as scope:
            loss= tf.nn.ctc_loss(self.ys,self.pred,self.length,ctc_merge_repeated=True)
            self.cost= tf.reduce_mean(loss)
            self.merge.append( tf.summary.scalar('cost',self.cost))

    def optimize(self):
            self.optimize=tf.train.AdamOptimizer(0.001,0.5,0.9).minimize(self.cost)
         

    def accuracy(self):
        with tf.name_scope('Accuracy') as scope:
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.pred,self.length)
            self.acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),self.ys))
            self.test_merge=tf.summary.scalar('Accuracy',self.acc)
