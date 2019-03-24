# Modified to be able to feed length 1 input
# and use present hidden state as the initial
#state of next input
#2017/5/4

import numpy as np
import copy
import matplotlib.pyplot as plt

############# Setting parameter #############

relative=True

samples=1
alldata=[]
alllength=np.zeros(samples,np.int32)



xlist=[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63]
ylist=[1,4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61,64]
zlist=[2,5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62,65]


all=999998
print('************** data{}'.format(all))
data=np.genfromtxt('file'+str(all)+'/joint.csv',delimiter=',')
spec=np.genfromtxt('file'+str(all)+'/spec.csv',delimiter=',')

length=int(spec[1])





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
    return  temp
    #########################

nortowrist()
if relative:
    inputdata = relative_track_batch(length)
else:
    inputdata = data


alllength[0]=length






np.save('DEMO3\data_relative.npy',inputdata)

np.save('DEMO3\length.npy',alllength)
