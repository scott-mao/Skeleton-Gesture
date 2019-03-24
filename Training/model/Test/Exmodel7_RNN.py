## This model is only for testing
#Seperete finger information
import tensorflow as tf
from model.op import *

class model(object):
    def __init__(self, feature_size, output_size, cell_size,layer,part,reuse):

        self.feature_size = feature_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.part = part
        self.layer = layer-1

        self.reuse=reuse
        self.outputalllayer=[]
        self.stateall=[]
        with tf.name_scope('input'):
            self.drop=tf.placeholder(tf.float32,name='Dropout_ratio')
            self.length = tf.placeholder(tf.int32, [None], name='length')  # length of each sequence
            self.input = tf.placeholder(tf.float32, [None, None, feature_size], name='X_in')
            self.ys = tf.sparse_placeholder(tf.int32, name='target')

        self.buildmodel()



    def buildmodel(self):
        if self.reuse ==False:#CPU original model/test model
            self.add_input()
            self.add_rnn_hidden()
            self.add_output_layer()
            self.accuracy()

        else:
            self.add_input()
            self.add_rnn_hidden()
            self.add_output_layer()

    def add_input(self):
        with tf.variable_scope('NN') as scope:
            if self.reuse:
                scope.reuse_variables()
            self.tempout = []
            input_part = [self.input[:, :, 0:3],
                          self.input[:, :, 3:6],
                          self.input[:, :, 6:18],
                          self.input[:, :, 18:30],
                          self.input[:, :, 30:42],
                          self.input[:, :, 42:54],
                          self.input[:, :, 54:66]]  # W C T I M R P

            for ind, part in enumerate(input_part):
                with tf.variable_scope('part' + str(ind)) as scope2:
                    if self.reuse:
                        scope2.reuse_variables()

                    shape = part.get_shape()
                    inshape = shape[2].value
                    outshape = self.part[ind]
                    cell=tf.nn.rnn_cell.GRUCell(outshape, reuse=False)
                    #cell = tf.nn.rnn_cell.BasicRNNCell(outshape, reuse=False)
                    output_last, state_last = tf.nn.dynamic_rnn(cell
                                                                , part
                                                                , dtype=tf.float32
                                                                , time_major=True
                                                                , sequence_length=self.length
                                                                , scope='RNNlayer')
                    output_last = tf.contrib.layers.layer_norm(output_last)
                    self.tempout.append(output_last)
            self.inputcombine = tf.concat(self.tempout, 2)


    def add_rnn_hidden(self):
        self.in_bn=[]
        with tf.variable_scope('RNN') as scope:
            if self.reuse :
                scope.reuse_variables()

            def lstml_cell():

                    return tf.contrib.rnn.LayerNormBasicLSTMCell(self.cell_size, reuse=False)

            def gru_cell():
                cell = tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=False)
                return cell
            output_last = self.inputcombine

            for i in range(self.layer):
                with tf.variable_scope('layer' + str(i)):
                    if i>0:
                        output_last = tf.contrib.layers.layer_norm(output_last)
                    shape = tf.shape(output_last)
                    cell=gru_cell()

                    output_last, state_last = tf.nn.dynamic_rnn(cell
                                                                , output_last
                                                                , dtype=tf.float32
                                                                , time_major=True

                                                                , sequence_length=self.length
                                                                , scope='RNNlayer')
                    self.outputalllayer.append(output_last)
                    self.stateall.append(state_last)
            self.output_last = output_last




    def add_output_layer(self):
        with tf.variable_scope('Output') as scope:
            if self.reuse:
                scope.reuse_variables()
            self.output_last = tf.contrib.layers.layer_norm(self.output_last)

            outall=tf.reshape(self.output_last,[-1,self.cell_size])
            Ws_out = weight_variable([self.cell_size, self.output_size],reuse=self.reuse)
            bs_out = bias_variable([self.output_size, ],reuse=self.reuse)
            with tf.name_scope('Wx_plus_b'):
                temp=tf.matmul(outall, Ws_out) + bs_out
				## comment here to disable normalization before softmax
                max = tf.reduce_max(temp, 1, keep_dims=True)
                temp=temp-max
				##
                shape=tf.shape(self.output_last)
                self.predall=tf.reshape(temp,[shape[0],shape[1],self.output_size])

    def accuracy(self):
        with tf.name_scope('Accuracy') as scope:

             self.outall=self.predall[:,:,0:14]
             self.outputall=tf.nn.softmax(logits=self.predall,name='Predition_labels')
             self.decode=tf.nn.ctc_beam_search_decoder(self.predall,self.length)
             self.loss=tf.nn.ctc_loss(self.ys,self.predall,self.length)