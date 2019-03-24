#Retreive Transfer value
import tensorflow as tf
import numpy as np
import model.Exmodel7_RNN as transfermodel
#import model.ctcmodel7_transfer_1RNN as transfermodel
import model.Exmodel6_transfer2 as EXmodel
import re
import os
from util import *
print(tf.__version__)
################ Hyper parameters ###############

MAX_test=150
FEATURE=np.array([66])
OUTPUT=15
CELL=np.array([100])
BATCH_test=140
BATCH_real=150
layer=2
layer_size=1
testCTC=True#False for preseg
draw=False
color=['xkcd:coral','xkcd:darkblue','xkcd:azure','xkcd:brown','xkcd:fuchsia','xkcd:lime','xkcd:olive'
    ,'xkcd:red','xkcd:cyan','xkcd:wheat','xkcd:purple','xkcd:crimson','xkcd:teal','xkcd:silver','xkcd:black']
#color=['xkcd:coral','xkcd:lime','xkcd:olive','xkcd:purple','xkcd:black']


############### Loading data ################
############### Loading data ################
datapath='./data/cont_data_6/'
Testing_in=np.load(datapath+'Testing_all_normalize_to_wrist_expand3_relativetrack.npy')
shape=np.shape(Testing_in)
MAX=int(shape[0])
Testing_label=np.load(datapath+'Testing_label_all_expand3.npy')
Testing_length=np.load(datapath+'Testing_length_all_expand3.npy')
Testing_labelnum=np.load(datapath+'Testing_label_all_number.npy')
Testing_section=np.load(datapath+'Testing_section.npy')
alltestlabel=np.sum(Testing_labelnum,dtype=np.int32)
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
test_target=tf.SparseTensorValue(indices=ind,values=value,dense_shape=[BATCH_test,10])

shouldfind=np.sum(Testing_labelnum)
#list=np.load(datapath+'train_list_expand34g.npy')
##############  Real case data ###########
##############  Real case data ###########

nodataset='validation'
folder='Realcase_new/'

Real_in=np.load('data/'+folder+nodataset+'_set/data_relative_sep.npy')
Real_label=np.load('data/'+folder+nodataset+'_set/label_sep.npy')
Real_length=np.load('data/'+folder+nodataset+'_set/length_sep.npy')
Real_labelnum=np.load('data/'+folder+nodataset+'_set/labelnum_sep.npy')
Real_section=np.load('data/'+folder+nodataset+'_set/labeltime_sep.npy')


shapereal=np.shape(Real_in)
MAXreal=int(shapereal[0])

allreallabel=np.sum(Real_labelnum,dtype=np.int32)
valuereal=np.zeros(allreallabel,np.int32)
indreal=np.zeros([allreallabel,2],np.int32)
now=0
for batch,(nowlabel,nowleng) in enumerate(zip(Real_label,Real_labelnum)):
    l=int(nowleng)
    indreal[now:now+l,0]=batch
    F=np.arange(l,dtype=np.int32)
    indreal[now:now+l,1]=np.arange(l,dtype=np.int32)
    valuereal[now:now+l]=nowlabel[0:l]
    now=now+l
Real_target=tf.SparseTensorValue(indices=indreal,values=valuereal,dense_shape=[BATCH_real,9])

shouldfind_real=np.sum(Real_labelnum)


##Testing data

Real_inT=np.load('data/'+folder+'testing_set/data_relative.npy')

Real_labelT=np.load('data/'+folder+'testing_set/label.npy')
Real_lengthT=np.load('data/'+folder+'testing_set/length.npy')
Real_labelnumT=np.load('data/'+folder+'testing_set/labelnum.npy')
Real_sectionT=np.load('data/'+folder+'testing_set/labeltime.npy')
'''
Real_inT=np.load('data/'+folder+'testing_set/Testing_all_normalize_to_wrist_expand3_relativetrack.npy')

Real_labelT=np.load('data/'+folder+'testing_set/Testing_label_all_expand3.npy')
Real_lengthT=np.load('data/'+folder+'testing_set/Testing_length_all_expand3.npy')
Real_labelnumT=np.load('data/'+folder+'testing_set/Testing_label_all_number.npy')
Real_sectionT=np.load('data/'+folder+'testing_set/Testing_section.npy')
shaperealT=np.shape(Real_inT)
MAXrealT=int(shaperealT[0])
'''
allreallabelT=np.sum(Real_labelnumT,dtype=np.int32)
valuerealT=np.zeros(allreallabelT,np.int32)
indrealT=np.zeros([allreallabelT,2],np.int32)
nowT=0
for batch,(nowlabel,nowleng) in enumerate(zip(Real_labelT,Real_labelnumT)):
    l=int(nowleng)
    indrealT[nowT:nowT+l,0]=batch

    indrealT[nowT:nowT+l,1]=np.arange(l,dtype=np.int32)
    valuerealT[nowT:nowT+l]=nowlabel[0:l]
    nowT=nowT+l
Real_targetT=tf.SparseTensorValue(indices=indrealT,values=valuerealT,dense_shape=[12,9])

shouldfind_realT=np.sum(Real_labelnumT)
##########################################
savedir = 'ctc_cont_exp_detail/grulnsection'+str(CELL[0])+"_l"+str(layer)+'/valid/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
tf.reset_default_graph()
with tf.Session() as sess:

        part = [15,15,15,15,15,15,15]
        #part = [3, 3, 6, 6, 6, 6, 6]


        resallx=["./ctc_cont_rand_finger/grulnv2100_l2/best-model-10600",
                "./ctc_cont_rand_finger/gruln100_l2/best-model-14100",
                "./ctc_cont_rand_finger/gruln2merge100_l2/my-model-6400",
                "./ctc_cont_new_randv2/gruln100_l2/my-model-14900",
                "./ctc_cont_rand_section/gruv4_alphavary100_l2/my-model-22300",
                "./ctc_cont_new_randv2/grulnmerge100_l2/my-model-14900",
                "./ctc_cont_new_randv2/grulnmerge100_l2/best-model-3000"]
        resall=[ "./ctc_cont_rand_section/gruv4_alphavary180_l2/my-model-14900",]
        resall1=["./ctc_cont_new_rand_data6/grulnmerge_noR100_l2/my-model-14900",
                 "./ctc_cont_new_rand_data6/grulnmerge_noR100_l2/best-model-6000",]
        resallx=["./ctc_cont_new_rand_data6/grulnmerge100_l2/my-model-14900",]
                #"./ctc_cont_new_rand_data6/grulnmerge100_l2/best-model-5800",
                 #]


        resall=["./ctc_cont_rand_finger_input/gruln2merge100_l2/best-model-2300",
                "./ctc_cont_rand_finger_input/gruln2merge100_l2/my-model-14900"]

        resall=["./ctc_cont_rand_finger_input/gruln100_l2/best-model-7100",
                "./ctc_cont_rand_input/grulnv2100_l2/best-model-8100",
                "./ctc_cont_rand_finger_input/gruln2merge2100_l2/my-model-14900",
                "./ctc_cont_rand_input/grumerge100_l2/my-model-14900",
                "./ctc_cont_rand_input/grumerge100_l2/best-model-1600"]
        resall=["./ctc_cont_rand_finger_input/grulnnosig100_l2/best-model-14700"]
        resall=["./ctc_cont_rand_section/grudata6_alphavary100_l2/my-model-14900"]
       # resallx=["./ctc_cont_new_rand_data6/grulninput100_l2/my-model-14900",]
        resall = ["./ctc_cont_new_rand_data6/grunofinallnnomax100_l2/my-model-11700",]
        resall=["./ctc_cont_new_rand_realcase/gruln100_l2/my-model-9900"]
        resall=["./ctc_cont_rand_section/grudata6_alphavary100_l2/my-model-14900"]
        resallx = ["./ctc_cont_new_rand_data6/grulnmerge100_l2/my-model-14900", ]
        resallx = ["./ctc_cont_new_rand_data_zeronormalize/grulnrnnin15100_l2/my-model-4700"]

        temp=np.zeros([7*3,10])
        counter=0
        for ind,res in enumerate(resallx):
            print(res)
            print('Getting Transfer Value...')
            testmodel = transfermodel.model(FEATURE, OUTPUT, CELL, layer, part, False)

            saver = tf.train.Saver()
            saver.restore(sess, res)
            var=tf.trainable_variables()
            feed_dict = {
                testmodel.input: Real_in,
                testmodel.length: Real_length,
               testmodel.ys: Real_target

            }
            transfervalue = sess.run(testmodel.output_last, feed_dict=feed_dict)
           # transfervalue=transfervalue[1]

            feed_dict = {
                testmodel.input: Real_inT,
                testmodel.length: Real_lengthT,
                testmodel.ys: Real_targetT

            }
            transfervalueT = sess.run(testmodel.output_last, feed_dict=feed_dict)
           # transfervalueT = transfervalueT[1]
            #eal_in = np.load('data/Realcase/' + nodataset + '_set/data_relativetrack_12.npy')
            np.save('data/'+folder+'/validation_set/transfer_value_beforeFCN_rnnin.npy',transfervalue)
            np.save('data/'+folder+'/testing_set/transfer_value_beforeFCN_rnnin.npy', transfervalueT)
            #np.save('data/'+folder+'/validation_set/transfer_value_beforefinalLN.npy',transfervalue)
            print("done.")
