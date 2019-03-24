import tensorflow as tf
import numpy as np
import model.ctcmodel8_rnn as mymodel
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

#setting 2:14 and uncomment 313~327 to enable the best Batch forcing method
BATCH=8#Real
BATCHrs=8#DHG cont and single
Rate=0.001
DROP=0.5
ITER=10000
NUM_GPU=1
#Train_par=0.9
#Test_par=0.1
datafolder='cont_data_zeronormalize'

realfolder='Realcase_new'
labelnummax=10

labelnummaxR=6
############### Loading data ################

Training_in=np.load('./data/'+datafolder+'/Training_all_normalize_to_wrist_expand3_relativetrack.npy')
Training_label=np.load('./data/'+datafolder+'/Training_label_all_expand3.npy')
Training_length=np.load('./data/'+datafolder+'/Training_length_all_expand3.npy')
Training_labelnum=np.load('./data/'+datafolder+'/Training_label_all_number.npy')



Training_inR=np.load('./data/'+realfolder+'/validation_set/data_relative_sep.npy')
Training_labelR=np.load('./data/'+realfolder+'/validation_set/label_sep.npy')
Training_lengthR=np.load('./data/'+realfolder+'/validation_set/length_sep.npy')
Training_labelnumR=np.load('./data/'+realfolder+'/validation_set/labelnum_sep.npy')
ind=np.arange(150,dtype=np.int32)
ind=np.random.permutation(ind)
Training_inR=Training_inR[:,ind[0:75],:]
Training_labelR=Training_labelR[ind[0:75],:]
Training_lengthR=Training_lengthR[ind[0:75]]
Training_labelnumR=Training_labelnumR[ind[0:75]]

shape=np.shape(Training_in)
MAXC=shape[0]
MAX_testC=shape[0]


shapeR=np.shape(Training_inR)
MAXR=shapeR[0]
MAX_testR=shapeR[0]

MAX=max(MAXC,MAXR)
MAX_test=MAX

datanum=shape[1]
minibatchnum=datanum//BATCHrs
allTrain = np.zeros([datanum, MAX, FEATURE[0] + labelnummax+2])  # 2 for length and label
allTrain[:,0 :MAXC, 0:FEATURE[0]] = np.transpose(Training_in,[1,0,2])
allTrain[:,0,FEATURE[0]]=Training_labelnum
allTrain[:, 0, FEATURE[0]+1:FEATURE[0]+labelnummax+1] = Training_label
allTrain[:, 0, FEATURE[0]+labelnummax+1] = Training_length

datanumR=shapeR[1]
minibatchnumR=datanumR//BATCH
allTrainR = np.zeros([datanumR, MAX, FEATURE[0] + labelnummaxR+2])  # 2 for length and label
allTrainR[:,0 :MAXR, 0:FEATURE[0]] = np.transpose(Training_inR,[1,0,2])
allTrainR[:,0,FEATURE[0]]=Training_labelnumR
allTrainR[:, 0, FEATURE[0]+1:FEATURE[0]+labelnummaxR+1] = Training_labelR
allTrainR[:, 0, FEATURE[0]+labelnummaxR+1] = Training_lengthR

Testing_in=np.load('./data/'+datafolder+'/Testing_all_normalize_to_wrist_expand3_relativetrack.npy')
Testing_label=np.load('./data/'+datafolder+'/Testing_label_all_expand3.npy')
Testing_length=np.load('./data/'+datafolder+'/Testing_length_all_expand3.npy')
Testing_labelnum=np.load('./data/'+datafolder+'/Testing_label_all_number.npy')

Testing_inR=np.load('./data/'+realfolder+'/testing_set/data_relative.npy')
Testing_labelR=np.load('./data/'+realfolder+'/testing_set/label.npy')
Testing_lengthR=np.load('./data/'+realfolder+'/testing_set/length.npy')
Testing_labelnumR=np.load('./data/'+realfolder+'/testing_set/labelnum.npy')

#Testing_inall = np.concatenate((Testing_in, Testing_inR), axis=1)
#Testing_lengthall = np.concatenate((Testing_length, Testing_lengthR))

shape=np.shape(Testing_in)
testnum=shape[1]

shapeR=np.shape(Testing_inR)
testnumR=shapeR[1]
## make sparse target

alltestlabel=np.sum(Testing_labelnum,dtype=np.int32)

alltestlabelR=np.sum(Testing_labelnumR,dtype=np.int32)


value=np.zeros(alltestlabel,np.int32)
ind=np.zeros([alltestlabel,2],np.int32)
now=0
for batch,(nowlabel,nowleng) in enumerate(zip(Testing_label,Testing_labelnum)):
    l=int(nowleng)
    ind[now:now+l,0]=batch
    F=np.arange(l,dtype=np.int32)
    ind[now:now+l,1]=np.arange(l,dtype=np.int32)
    value[now:now+l]=nowlabel[0:l]
    now=now+l
test_target=tf.SparseTensorValue(indices=ind,values=value,dense_shape=[testnum,labelnummax])




value=np.zeros(alltestlabelR,np.int32)
ind=np.zeros([alltestlabelR,2],np.int32)
now=0
for batch,(nowlabel,nowleng) in enumerate(zip(Testing_labelR,Testing_labelnumR)):
    l=int(nowleng)
    ind[now:now+l,0]=batch
    F=np.arange(l,dtype=np.int32)
    ind[now:now+l,1]=np.arange(l,dtype=np.int32)
    value[now:now+l]=nowlabel[0:l]
    now=now+l

test_targetR = tf.SparseTensorValue(indices=ind, values=value, dense_shape=[testnumR, labelnummaxR])


    ############### Creating special matrix ########
for CELL in CELL_all:
  for layer in Layer_all:
    savedir='Batchtranfer/zerostart2'
    tf.reset_default_graph()


    print(CELL,savedir)

    with tf.name_scope('Training_model'):
        model = mymodel.model(FEATURE, OUTPUT, CELL,layer,inputpart,False)
    with tf.name_scope('Testing_model'):
        testmodel=mymodel.model(FEATURE, OUTPUT, CELL,layer,inputpart,True)
    with tf.name_scope('Testing_modelR'):
        testmodelR=mymodel.model(FEATURE, OUTPUT, CELL,layer,inputpart,True)

    totalcostsum=tf.summary.scalar('Testing_totalloss',testmodel.cost+testmodelR.cost)
    #config = tf.ConfigProto(allow_soft_placement=True)
    best=100
    lastcost=30

    with tf.Session() as sess:
        saverbest=tf.train.Saver(max_to_keep=5)
        var=tf.trainable_variables()
        saver=tf.train.Saver(max_to_keep=100,var_list=var)

       # D=sess.run(var)


        writer = tf.summary.FileWriter(savedir, sess.graph)
        #assert 1==2
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess,savedir+'/ori-model-7800')
        #saver.restore(sess, "saver/my-model-400")
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
        startiS = 0
        startiR=0
        '''
        A=np.arange(BATCH,dtype=np.int32)
        A=np.tile(A,[labelnum,1])
        A=np.reshape(np.transpose(A),[BATCH*labelnum,1])
        B=np.arange(labelnum,dtype=np.int32)
        B=np.transpose(np.tile(B,[1,BATCH]))
        index=np.zeros([BATCH*labelnum,2],dtype=np.int32)
        index[:,0]=A[:,0]
        index[:,1]=B[:,0]
        '''
        for iteration in range(0+counter,ITER+counter,NUM_GPU):

            batch = iteration % minibatchnum
            if iteration == 0 or batch == 0 or starti == 0:
                new = shuffle(allTrain, 'Train')
                datain = new[:, :, 0:FEATURE[0]]
                datain = np.transpose(datain, [1, 0, 2])
                labelnum=np.int32(new[:, 0, FEATURE[0]])
                datalabel = np.int32(new[:, 0, FEATURE[0]+1:FEATURE[0]+labelnummax+1])
                datalength = np.int32((new[:, 0, FEATURE[0] + labelnummax+1]))
                starti = 1


            batchR = iteration % minibatchnumR
            if iteration == 0 or batchR == 0 or startiR == 0:
                newR = shuffle(allTrainR, 'TrainR')
                datainR = newR[:, :, 0:FEATURE[0]]
                datainR = np.transpose(datainR, [1, 0, 2])
                labelnumR=np.int32(newR[:, 0, FEATURE[0]])
                datalabelR = np.int32(newR[:, 0, FEATURE[0]+1:FEATURE[0]+labelnummaxR+1])
                datalengthR = np.int32((newR[:, 0, FEATURE[0] + labelnummaxR+1]))
                startiR = 1


            feed_dict={}

            ######### feed different batch data to models ########

            start = (batch ) * BATCHrs
            end = (batch  + 1) * BATCHrs

            startR = (batchR) * BATCH
            endR = (batchR + 1) * BATCH


            labeltemp=datalabel[start:end,:]
            labelnumtemp=labelnum[start:end]
            alltrainlabel=np.sum(labelnumtemp)



            labeltempR = datalabelR[startR:endR, :]
            labelnumtempR = labelnumR[startR:endR]
            alltrainlabelR = np.sum(labelnumtempR)

            finaltrain=alltrainlabel+alltrainlabelR

            value = np.zeros(finaltrain, np.int32)
            ind = np.zeros([finaltrain, 2], np.int32)
            now=0
            for batch, (nowlabel, nowleng) in enumerate(zip(labeltemp, labelnumtemp)):
                l = int(nowleng)
                ind[now:now + l, 0] = batch
                F = np.arange(l, dtype=np.int32)
                ind[now:now + l, 1] = np.arange(l, dtype=np.int32)
                value[now:now + l] = nowlabel[0:l]
                now = now + l



            for batch, (nowlabel, nowleng) in enumerate(zip(labeltempR, labelnumtempR)):
                l = int(nowleng)
                ind[now:now + l, 0] = batch+BATCHrs
                F = np.arange(l, dtype=np.int32)
                ind[now:now + l, 1] = np.arange(l, dtype=np.int32)
                value[now:now + l] = nowlabel[0:l]
                now = now + l

            train_target = tf.SparseTensorValue(indices=ind, values=value, dense_shape=[BATCH*2, labelnummax])

            #labeltemp=np.transpose(labeltemp)
           # labeltemp=np.reshape(labeltemp,[labelnum*BATCH])
            #label = tf.SparseTensorValue(index,labeltemp,[BATCH,labelnum])
            dict= {
            model.input:np.concatenate((datain[:,start:end,:],datainR[:,startR:endR,:]),axis=1),
            model.length:np.concatenate((datalength[start:end],datalengthR[startR:endR])),
            model.ys:train_target,

            }

            feed_dict.update(dict)




            _, loss,summary= sess.run([model.optimize,model.cost,model.merge],feed_dict=feed_dict)

            print('iteration:{0}, Loss:{1}'.format(iteration,loss))

            writer.add_summary(summary, iteration)
			#uncomment to enable changing ratio of the data in a batch
            '''
            if loss<lastcost//2 :

                BATCH+=2
                BATCHrs-=2
                if BATCH>8:
                    BATCH=8
                if BATCHrs<8:
                    BATCHrs=8
                minibatchnum = datanum // BATCHrs
                minibatchnumR = datanumR // BATCH
                lastcost=loss
                print(BATCH)
                print(BATCHrs)
            '''
            if iteration%100==0:
                batch=0
                print('Save model to:saver/my-model-{}'.format(iteration))
                saver.save(sess, savedir+'/my-model', global_step=iteration)

                feed_dict1 = {
                    testmodel.input: Testing_in,
                    testmodel.ys:test_target,
                    testmodel.length: Testing_length,

                    }


                out,testa,losstest,lossnum=sess.run([testmodel.acc,testmodel.test_merge, testmodel.costsum,testmodel.cost],feed_dict=feed_dict1)
                print(out)

                writer.add_summary(testa,iteration)
                writer.add_summary(losstest,iteration)


                ################ REAL CASE #################
                feed_dict = {
                    testmodelR.input: Testing_inR,
                    testmodelR.ys: test_targetR,
                    testmodelR.length: Testing_lengthR,
                    
                }



                out, testa, losstest, lossnumR = sess.run(
                    [testmodelR.acc, testmodelR.test_merge, testmodelR.costsum, testmodelR.cost], feed_dict=feed_dict)
                print(out)


                feed_dict1.update(feed_dict)
                writer.add_summary(testa, iteration)
                writer.add_summary(losstest, iteration)

                total=sess.run(totalcostsum,feed_dict1)
                writer.add_summary(total,iteration)

                if (lossnum+lossnumR)/2<best:
                    saverbest.save(sess, savedir+'/best-model', global_step=iteration)
                    best=lossnum
