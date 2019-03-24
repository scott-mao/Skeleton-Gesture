#Segmented gesture recognition
import tensorflow as tf
import numpy as np
import model.noseperate7 as mymodel
from util import *
import re

print(tf.__version__)
################ Hyper parameters ###############
expandall=[]
#expandall.append(1)
#expandall.append(2)
expandall.append(3)
#expandall.append(4)
#expandall.append(5)

MAX_all=[]
#MAX_all.append(150)
#MAX_all.append(250)
MAX_all.append(150)
#MAX_all.append(280)
#MAX_all.append(280)

MAX_test_all=[]
#MAX_test_all.append(150)
#MAX_test_all.append(250)
MAX_test_all.append(150)
#MAX_test_all.append(280)
#MAX_test_all.append(280)

FEATURE=np.array([66])
OUTPUT=14
CELL=100
#layer=2
CELL_all=[]
#CELL_all.append(50)
CELL_all.append(100)
##CELL_all.append(200)
layer_size=1
Layer_all=[]
#Layer_all.append(1)
Layer_all.append(2)
#Layer_all.append(3)
#Layer_all.append(4)
BATCH=16

LR=0.001
DROP=0.5
ITER=1
NUM_GPU=1
Train_par=0.8
Test_par=0.2
for expand,MAX,MAX_test in zip(expandall,MAX_all,MAX_test_all):
    BATCH_test=int(2800*Test_par)

    Total = int(2800*Train_par * expand)
    minibatch_num=int(Total/BATCH)
    ############### Loading data ################
    folder='nzsnze_essai2'
    Training_in=np.load('./data/'+folder+'/Training_all_normalize_to_wrist_expand'+str(expand)+'.npy')
    Training_label=np.load('./data/'+folder+'/Training_label_all_expand'+str(expand)+'.npy')
    Training_length=np.load('./data/'+folder+'/Training_length_all_expand'+str(expand)+'.npy')
    # Training_length=np.int32(Training_length)
    Training_in=np.transpose(Training_in,[1,0,2])
    allTrain = np.zeros([Total, MAX, FEATURE[0] + 2])  # 2 for length and label
    allTrain[:, :, 0:FEATURE[0]] = Training_in
    allTrain[:, 0, FEATURE[0]] = Training_label
    allTrain[:, 0, FEATURE[0] + 1] = Training_length

    Testing_in=np.load('./data/'+folder+'/Testing_all_normalize_to_wrist_expand'+str(expand)+'.npy')
    Testing_label=np.load('./data/'+folder+'/Testing_label_all_expand'+str(expand)+'.npy')
    Testing_length=np.load('./data/'+folder+'/Testing_length_all_expand'+str(expand)+'.npy')
        ############### Creating special matrix ########
    for CELL in CELL_all:
      for layer in Layer_all:
        savedir='./preseg/gruabs'+str(CELL)+'_l'+str(layer)
        tf.reset_default_graph()


        print(CELL,savedir)
        testlengthmat=np.zeros((MAX_test,BATCH_test,CELL),dtype=float)


        for sequence in range(BATCH_test):
                  testlengthmat[Testing_length[sequence]-1,sequence,:]=np.ones(CELL,dtype=float)
        ################### GRAD ##################
        def average_gradients(grads):#grads:[[grad0, grad1,..], [grad0,grad1,..]..]
          averaged_grads = []
          for grads_per_var in zip(*grads):
            grads = tf.reduce_mean(grads_per_var, 0)
            averaged_grads.append(grads)
          return averaged_grads
        ###################  Main Variable in CPU  #################

        with tf.device("cpu:0"):
            with tf.name_scope('CPU_original'):
              cpumodel=mymodel.model(FEATURE, OUTPUT, CELL,layer,False)
        ###################  Setting multiple GPUs #################
        grads=[]
        for i in range(NUM_GPU):
            with tf.device("/gpu:%d"%i):
              with tf.name_scope("tower_%d"%i):
                model = mymodel.model(FEATURE, OUTPUT, CELL,layer,True)

                tf.add_to_collection("train_model", model)
                grads.append(model.grads) #grad 是通过tf.gradients(loss, vars)求得
                #以下这些add_to_collection可以直接在模型内部完成。
                # 将loss放到 collection中， 方便以后操作
                tf.add_to_collection("loss",model.cost)

                #将 summary.merge op放到collection中，方便操作
                tf.add_to_collection("merge_summary", model.merge)

        with tf.device("cpu:0"):
            with tf.name_scope('Train'):
                avg_gradient = average_gradients(grads)# average_gradients后面说明
                #capped_gvs = [(tf.clip_by_value(grad,clip_value_max=5.0,clip_value_min=-5.0)) for grad in avg_gradient]
                opt = tf.train.AdamOptimizer(LR)
                train_op=opt.apply_gradients(zip(avg_gradient,tf.trainable_variables()))
        ####################### Setting Testing Net ##############3
            ############## training ############
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=config) as sess:

            saver=tf.train.Saver(max_to_keep=10)
            writer = tf.summary.FileWriter(savedir, sess.graph)

            sess.run(tf.global_variables_initializer())
            #saver.restore(sess, "saver/my-model-400")
            ckpt = tf.train.get_checkpoint_state(savedir)
            counter=0
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(savedir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            else:
                print('fail to load')
            datain=[]
            datalabel=[]
            datalength=[]
            starti=0
            for iteration in range(0+counter,ITER+counter,NUM_GPU):

                batch=iteration%minibatch_num
                if iteration==0 or batch==0 or starti==0:
                    new = shuffle(allTrain, 'Train')
                    datain = new[:, :, 0:FEATURE[0]]
                    datain = np.transpose(datain, [1, 0, 2])
                    datalabel = (new[:, 0, FEATURE[0]])
                    datalength = np.int32((new[:, 0, FEATURE[0] + 1]))
                    starti=1


                models=tf.get_collection("train_model")
                merges=tf.get_collection("merge_summary")
                losses=tf.get_collection("loss")
                feed_dict={}
                trainlengthmat=np.zeros([MAX,BATCH,CELL])
                ######### feed different batch data to models ########
                for id,modeln in enumerate(models):
                    start=(batch+id)*BATCH
                    end=(batch+id+1)*BATCH
                    trainlengthmat[datalength[start:end] - 1, np.arange(0,BATCH), :] = np.ones(CELL,dtype=float)


                    dict= {
                    modeln.input:datain[:,start:end,:],
                    modeln.length:datalength[start:end],
                    modeln.ys:datalabel[start:end],
                    modeln.lengthmat:trainlengthmat,
                    modeln.drop:0.5
                    }
                    feed_dict.update(dict)


                _, loss,summary= sess.run([train_op,losses,merges],feed_dict=feed_dict)
                print('iteration:{0}, Loss:{1}'.format(iteration,loss))
                for sum in summary:
                  writer.add_summary(sum, iteration)

                if iteration%100==0:
                    batch=0
                    print('Save model to:saver/my-model-{}'.format(iteration))
                    saver.save(sess, savedir+'/my-model', global_step=iteration)

                    feed_dict = {
                        cpumodel.input: Testing_in,
                        cpumodel.drop:1.0,
                        cpumodel.length: Testing_length,
                        cpumodel.ys: Testing_label,
                        cpumodel.lengthmat: testlengthmat}

                    summary, accuracy = sess.run([cpumodel.merge, cpumodel.accuracy], feed_dict=feed_dict)
                    print('iteration:{0}, Testing Accuracy:{1}'.format(iteration,accuracy))
                    writer.add_summary(summary, iteration)



