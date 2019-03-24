#Transform recoding data (.csv) into .npy
#normalize with respect to wrist

#2017/5/4

import numpy as np
import copy
import matplotlib.pyplot as plt

############# Setting parameter #############

relative=False#relative replacement or not
center=True#use center as track or not, if not, use wrist as track
samples=100
alldata=[]
alllength=np.zeros(samples,np.int32)
alllabel=[]
alllabeltime=[]
alllabelnum=np.zeros(samples,np.int32)
label_correct=np.load('labellist.npy')
MAX=-1
labelMAX=0
K=np.zeros(982)
xlist=[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63]
ylist=[1,4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61,64]
zlist=[2,5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62,65]

for all in range(samples):
     if all ==100:
         continue
     else:
        print('************** data{}'.format(all+1))
        data=np.genfromtxt('file'+str(all+1)+'/joint.csv',delimiter=',')
        spec=np.genfromtxt('file'+str(all+1)+'/spec.csv',delimiter=',')
        time=np.genfromtxt('file'+str(all+1)+'/section.csv',delimiter=',')
        length=int(spec[1])
        numlabel=int(spec[2])
        alllabelnum[all]=numlabel
        label=np.zeros(numlabel,np.int32)
        labeltime=np.zeros([numlabel,2],np.int32)
        for i in range(numlabel):
            detect=label_correct[all,i]

            if detect ==12:
                detect=11
            elif detect ==11:
                detect=12
            else:
                detect=detect
            label[i]=np.int32(detect)
            labeltime[i,:]=np.int32(time[i,:])#start and end

        if length>MAX:
            MAX=length
        K[int(length)]+=1
        if numlabel>labelMAX:
            labelMAX=numlabel


        def nortowrist():
            ###### normalize to wrist
            xc = copy.copy(data[:, 0])
            yc = copy.copy(data[:, 1])
            zc = copy.copy(data[:, 2])
            data[:, [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63]] -= np.tile(
                np.reshape(xc, [-1, 1]), (1, 21))
            data[:, [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64]] -= np.tile(
                np.reshape(yc, [-1, 1]), (1, 21))
            data[:, [5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65]] -= np.tile(
                np.reshape(zc, [-1, 1]), (1, 21))

        def relative_track_batch(MAX):
            ### do this every iteration to make first step replacement=0
            temp = copy.copy(data[:, 0:66])
            lastplace = np.zeros([ MAX, 1])
            # x
            lastplace[1: MAX, 0] = temp[0: MAX - 1, 0]
            lastplace[0, 0] = temp[0, 0]
            temp[0: MAX, 0] -= lastplace[:, 0]
            # y
            lastplace[1: MAX, 0] = temp[0: MAX - 1, 1]
            lastplace[0, 0] = temp[0, 1]
            temp[0: MAX, 1] -= lastplace[:, 0]
            # z
            lastplace[1: MAX, 0] = temp[0: MAX - 1, 2]
            lastplace[0, 0] = temp[0, 2]
            temp[0: MAX, 2] -= lastplace[:, 0]
            temp = np.reshape(temp, [MAX, 1, 66])
            return temp


        def relative_track_center(now_length,center):
            ### do this every iteration to make first step replacement=0
            temp = copy.copy(data[:, 0:66])
            lastplace = np.zeros([now_length, 1])

            # x
            lastplace[1:now_length, 0] = copy.copy(centertrack[0:now_length - 1, 0])
            lastplace[0, 0] = copy.copy(centertrack[0, 0])
            temp[0:now_length, 0] = centertrack[:, 0] - lastplace[:, 0]
            # y
            lastplace[1:now_length, 0] = copy.copy(centertrack[0:now_length - 1, 1])
            lastplace[0, 0] = copy.copy(centertrack[0, 1])
            temp[0:now_length, 1] = centertrack[:, 1] - lastplace[:, 0]
            # z
            lastplace[1:now_length, 0] = copy.copy(centertrack[0:now_length - 1, 2])
            lastplace[0, 0] = copy.copy(centertrack[0, 2])
            temp[0:now_length, 2] = centertrack[:, 2] - lastplace[:, 0]
            temp = np.reshape(temp, [now_length, 1, 66])
            return temp

        def allrelative(now_length):
            temp = copy.copy(data[:, 0:66])
            lastplace = np.zeros([now_length, 22])
            # x
            lastplace[1:now_length, :] = temp[0:now_length - 1, xlist]
            lastplace[0, :] = temp[0, xlist]
            temp[0:now_length, xlist] -= lastplace[:, :]
            # y
            lastplace[1:now_length, :] = temp[0:now_length - 1, ylist]
            lastplace[0, :] = temp[0, ylist]
            temp[0:now_length, ylist] -= lastplace[:, :]
            # z
            lastplace[1:now_length, :] = temp[0:now_length - 1, zlist]
            lastplace[0, :] = temp[0, zlist]
            temp[0:now_length, zlist] -= lastplace[:, :]
            return temp
            #########################
        if relative:
            if center:
                centertrack=copy.copy(data[:,3:6])
                nortowrist()
                inputdata=relative_track_center(length,centertrack)
            else:
                nortowrist()
                inputdata=relative_track_batch(length)
        else:
            nortowrist()
            inputdata=data
        alldata.append(inputdata)
        alllabel.append(label)
        alllength[all]=length
        alllabeltime.append(labeltime)

datanp=np.zeros([MAX,samples,66])
labelnp=np.zeros([samples,labelMAX],dtype=np.int32)
labelnumnp=np.zeros([samples,labelMAX,2],dtype=np.int32)



for ind,(datas,labels,labelt) in enumerate(zip(alldata,alllabel,alllabeltime)):
    shapedatas=np.shape(datas)
    ldatas=shapedatas[0]
    datanp[0:ldatas,ind,:]=np.reshape(datas,[ldatas,66])
    shapelabels=np.shape(labels)
    llabels=shapelabels[0]
    #labelnp[ind,0]=llabels
    #labelnumnp[ind,0,0]=llabels
    labelnp[ind,0:llabels]=labels
    labelnumnp[ind,0:llabels]=labelt


print(MAX)
#print(K)
#np.savetxt("K.csv",K,delimiter=',')
np.save('Realcase_abs\data_abs.npy',datanp)
np.save('Realcase_abs\label.npy',labelnp)
np.save('Realcase_abs\length.npy',alllength)
np.save('Realcase_abs\labeltime.npy',labelnumnp)
np.save('Realcase_abs\labelnum.npy',alllabelnum)