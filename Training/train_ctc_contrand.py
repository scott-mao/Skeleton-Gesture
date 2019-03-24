#use connecting DHG to train model using CTC
import tensorflow as tf
import numpy as np
import model.ctcmodel7 as mymodel
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
MAX_all.append(472)
#MAX_all.append(250)
#MAX_all.append(280)
#MAX_all.append(280)
#MAX_all.append(280)

MAX_test_all=[]
MAX_test_all.append(472)
#MAX_test_all.append(250)
#MAX_test_all.append(280)
#MAX_test_all.append(280)
#MAX_test_all.append(280)
inputpart=[15,15,15,15,15,15,15]#W C T I M R P
FEATURE=np.array([66])
encode=40
OUTPUT=15
CELL_all=[]
#CELL_all.append(np.array([20]))
#CELL_all.append(np.array([50]))
CELL_all.append(np.array([100]))
#CELL_all.append(np.array([150]))
#CELL_all.append(np.array([200]))
layer_size=1
Layer_all=[]
#Layer_all.append(2)
Layer_all.append(2)
#Layer_all.append(4)

BATCH=16

Rate=0.001
DROP=0.5
ITER=20000
NUM_GPU=1
#Train_par=0.9
#Test_par=0.1
datafolder='nzsnze_essai'
labelnummax=1

############### Loading data ################
#Training_labelnum=np.load('./data/'+datafolder+'/Training_label_all_number.npy')
Training_in=np.load('./data/'+datafolder+'/Training_all_normalize_to_wrist_expand3_relativetrackm9.npy')
Training_label=np.load('./data/'+datafolder+'/Training_label_all_expand3m9.npy')
Training_length=np.load('./data/'+datafolder+'/Training_length_all_expand3m9.npy')
#Training_labelnum=np.load('./data/'+datafolder+'/Training_label_all_number.npy')
shape=np.shape(Training_in)
MAX=shape[0]
MAX_test=shape[0]
datanum=shape[1]
Training_labelnum=np.ones(datanum,np.int64)
minibatchnum=datanum//BATCH
allTrain = np.zeros([datanum, MAX, FEATURE[0] + labelnummax+2])  # 2 for length and label
allTrain[:, :, 0:FEATURE[0]] = np.transpose(Training_in,[1,0,2])
allTrain[:,0,FEATURE[0]]=Training_labelnum
allTrain[:, 0, FEATURE[0]+1:FEATURE[0]+labelnummax+1] = np.reshape(Training_label,[-1,1])
allTrain[:, 0, FEATURE[0]+labelnummax+1] = Training_length


Testing_in=np.load('./data/'+datafolder+'/Testing_all_normalize_to_wrist_expand3_relativetrackm1.npy')
Testing_label=np.load('./data/'+datafolder+'/Testing_label_all_expand3m1.npy')
Testing_length=np.load('./data/'+datafolder+'/Testing_length_all_expand3m1.npy')
#Testing_labelnum=np.load('./data/'+datafolder+'/Testing_label_all_number.npy')

shape=np.shape(Testing_in)
testnum=shape[1]
Testing_labelnum=np.ones(testnum,np.int64)
## make sparse target

'''
value=np.zeros(datanum,np.int32)
ind=np.zeros([datanum,2],np.int32)
now=0
for i in range(datanum):
    l=3
    ind[now:now+l,0]=i
    ind[now:now+l,1]=np.arange(l,dtype=np.int32)
    value[now:now+l]=Training_label[i,0:l]
    now=now+l
target=tf.SparseTensorValue(indices=ind,values=value,dense_shape=[datanum,MAXlabel])
'''
alltestlabel=np.sum(Testing_labelnum,dtype=np.int32)
value=np.zeros(alltestlabel,np.int32)
ind=np.zeros([alltestlabel,2],np.int32)
now=0
for batch,(nowlabel,nowleng) in enumerate(zip(Testing_label,Testing_labelnum)):
    l=int(nowleng)
    ind[now:now+l,0]=batch
    F=np.arange(l,dtype=np.int32)
    ind[now:now+l,1]=np.arange(l,dtype=np.int32)
    #value[now:now+l]=nowlabel[0:l]
    value[now:now + l] = nowlabel
    now=now+l
test_target=tf.SparseTensorValue(indices=ind,values=value,dense_shape=[testnum,labelnummax])


  ############### Creating special matrix ########
for CELL in CELL_all:
  for layer in Layer_all:
    savedir='ctc_single/grunoln'+str(CELL[0])+'_l'+str(layer)
    tf.reset_default_graph()


    print(CELL,savedir)

    with tf.name_scope('Training_model'):
        model = mymodel.model(FEATURE, OUTPUT, CELL,layer,False)
    with tf.name_scope('Testing_model'):
        testmodel=mymodel.model(FEATURE, OUTPUT, CELL,layer,True)

    #config = tf.ConfigProto(allow_soft_placement=True)
    best=100
    with tf.Session() as sess:
        saverbest=tf.train.Saver(max_to_keep=5)
        saver=tf.train.Saver(max_to_keep=40)
        writer = tf.summary.FileWriter(savedir, sess.graph)
      
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(savedir)
        counter=0

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(savedir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print('loaded checkpoint successfully')
        else:
            print('fail to load')


        starti=0
       
        for iteration in range(0+counter,ITER+counter,NUM_GPU):

            batch = iteration % minibatchnum
			#shuffle the data
            if iteration == 0 or batch == 0 or starti == 0:
                new = shuffle(allTrain, 'Train')
                datain = new[:, :, 0:FEATURE[0]]
                datain = np.transpose(datain, [1, 0, 2])
                labelnum=np.int32(new[:, 0, FEATURE[0]])
                datalabel = np.int32(new[:, 0, FEATURE[0]+1:FEATURE[0]+labelnummax+1])
                datalength = np.int32((new[:, 0, FEATURE[0] + labelnummax+1]))
                starti = 1

            feed_dict={}

            ######### feed different batch data to models ########

            start = (batch ) * BATCH
            end = (batch  + 1) * BATCH

            labeltemp=datalabel[start:end,:]
            labelnumtemp=labelnum[start:end]
            alltrainlabel=np.sum(labelnumtemp)
            value = np.zeros(alltrainlabel, np.int32)
            ind = np.zeros([alltrainlabel, 2], np.int32)
            now=0
            for batch, (nowlabel, nowleng) in enumerate(zip(labeltemp, labelnumtemp)):
                l = int(nowleng)
                ind[now:now + l, 0] = batch
                F = np.arange(l, dtype=np.int32)
                ind[now:now + l, 1] = np.arange(l, dtype=np.int32)
                value[now:now + l] = nowlabel[0:l]
                now = now + l
            train_target = tf.SparseTensorValue(indices=ind, values=value, dense_shape=[BATCH, labelnummax])

            #labeltemp=np.transpose(labeltemp)
           # labeltemp=np.reshape(labeltemp,[labelnum*BATCH])
            #label = tf.SparseTensorValue(index,labeltemp,[BATCH,labelnum])

            dict= {
            model.input:datain[:,start:end,:],
            model.length:datalength[start:end],
            model.ys:train_target,
            model.drop:DROP
            }

            feed_dict.update(dict)




            _, loss,summary= sess.run([model.optimize,model.cost,model.merge],feed_dict=feed_dict)
            print('iteration:{0}, Loss:{1}'.format(iteration,loss))

            writer.add_summary(summary, iteration)

            if iteration%100==0:
                batch=0
                print('Save model to:saver/my-model-{}'.format(iteration))
                saver.save(sess, savedir+'/my-model', global_step=iteration)

                feed_dict = {
                    testmodel.input: Testing_in,
                    testmodel.ys:test_target,
                    testmodel.length: Testing_length,
                    testmodel.drop:1.0
                    }

                out,testa,losstest,lossnum=sess.run([testmodel.acc,testmodel.test_merge, testmodel.costsum,testmodel.cost],feed_dict=feed_dict)
                print(out)
                
                writer.add_summary(testa,iteration)
                writer.add_summary(losstest,iteration)
                if lossnum<best:
                    saverbest.save(sess, savedir+'/best-model', global_step=iteration)
                    best=lossnum
