#Fine tune the CTC model
#Tune the last RNN + Layer normalization + Fully connected layer
import tensorflow as tf
from model.op import *

class model(object):
    def __init__(self, feature_size, output_size, cell_size,layer,reuse):

        self.feature_size = feature_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.outputall=[]
        self.layer = 1#tune the last RNN only
        self.merge=[]
        self.reuse=reuse
        with tf.name_scope('input'):

            self.length = tf.placeholder(tf.int32, [None], name='length')  # length of each sequence
            self.input = tf.placeholder(tf.float32, [None, None, feature_size], name='transfer_in')
            self.ys = tf.sparse_placeholder(tf.int32, name='target')  # ys need to be one-hot

        self.buildmodel()
        if self.reuse == False:

            self.merge=tf.summary.merge(self.merge)



    def buildmodel(self):
        if self.reuse ==False:#CPU original model/test model
           
            self.add_rnn_hidden()
            self.add_output_layer()
            self.ctc_loss()
            self.optimize()

        else:
            
            self.add_rnn_hidden()
            self.add_output_layer()
            self.accuracy()
            self.ctc_loss()
            self.costsum=tf.summary.scalar('tcost',self.cost)
            


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

            output_last=self.input
            for i in range(self.layer):
               with tf.variable_scope('layer'+str(i)):
               # if i>0:
               #  output_last = tf.contrib.layers.layer_norm(output_last)

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
            #self.output_last = tf.contrib.layers.layer_norm(self.output_last, center=False)

            outall = tf.reshape(self.output_last, [-1, self.cell_size])
            with tf.name_scope('Wx_plus_b'):
                Ws_out = weight_variable([self.cell_size, self.output_size], reuse=self.reuse)
                bs_out = bias_variable([self.output_size, ], reuse=self.reuse)
                temp = tf.matmul(outall, Ws_out) + bs_out
				#uncommen to enable normalization before softmax
                #max=tf.reduce_max(temp,1,keep_dims=True)
                #temp=temp-max
                shape = tf.shape(self.output_last)
                self.pred = tf.reshape(temp, [shape[0], shape[1], self.output_size])

    def ctc_loss(self):
        with tf.name_scope('Loss') as scope:
            loss= tf.nn.ctc_loss(self.ys,self.pred,self.length,ctc_merge_repeated=True)
            self.cost= tf.reduce_mean(loss)
            self.merge.append( tf.summary.scalar('cost',self.cost))

    def optimize(self):
        with tf.name_scope('Optimizer'):
            #tvar=tf.trainable_variables()
           # self.varlist= [var for var in tvar if 'Wx_plus_b' in var.name]
           # self.varlist_fix=[var for var in tvar if 'Wx_plus_b' not in var.name]
           # grads=tf.gradients(self.cost,self.varlist,name='FCN_grad')
           # opt=tf.train.AdamOptimizer(0.001)
            self.optimize = tf.train.AdamOptimizer(0.001).minimize(self.cost)

            #self.optimize=opt.apply_gradients(zip(grads,self.varlist),name='optimizer')
            #self.optimize =  tf.train.MomentumOptimizer(0.01,0.95).minimize(self.cost)


    def accuracy(self):
        with tf.name_scope('Accuracy') as scope:
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.pred,self.length)
            self.acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),self.ys))
            self.test_merge=tf.summary.scalar('Accuracy',self.acc)
            self.outputall = tf.nn.softmax(logits=self.pred, name='Predition_labels')
