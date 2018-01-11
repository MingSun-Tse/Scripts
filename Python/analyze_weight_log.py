from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
import time

def get_index(log_i):
    '''
        return example: {"0":[],"1":[], "2": []}   
    '''
    lines = [l.strip() for l in open(log_i) if l.strip()]
    index = {}
    cnt = 0
    for l in lines:
        index[str(cnt)] = [int(i) for i in l.split(" ")]
        cnt += 1
    return index

    
def get_weight(log_w):
    '''
        return example: {"0":[[], [], []],"1":[[], [], []], "2": [[], [], []]}   
    '''
    lines = [l.strip() for l in open(log_w)]
    weight = {}
    cnt = 0
    for l in lines:
        if l:
            if str(cnt) in weight.keys():
                weight[str(cnt)].append([float(i) for i in l.split(" ")])
            else:
                weight[str(cnt)] = [[float(i) for i in l.split(" ")]]
        else:
            cnt += 1
    return weight

def plot_sub(l, index, w, d):
    iter_range = [10000, 11000]
    num_show = 3

    assert len(w) == len(d)
    num = len(w) ## num of weights to track
    plt.figure(figsize = (25, 10))
    CM = np.array([plt.cm.gist_ncar(i) for i in np.linspace(0, 0.9, num)])[np.random.permutation(num)]
    rand = np.random.permutation(num)
    index, w, d = np.array(index)[rand], np.array(w)[rand], np.array(d)[rand]
    for i in range(num_show): 
        print ("ploting weight #%s of layer %s..." % (i, l))
        plt.plot(smooth(d[i]), label = "d-" + str(index[i]), color = CM[::-1][i]) #color = CM[i]
        plt.plot(w[i], label = "w-" + str(index[i]), marker = ".", color = CM[::-1][i])
        plt.scatter(difference(w[i])[:len(d[i])], difference(d[i]), label = "d-w" + str(index[i]))
        
    plt.title("Change of weight & diff in layer-%s" % l, fontsize = 25)
    plt.xlabel("Iter", fontsize = 20); plt.ylabel("Weight or Diff Value", fontsize = 20)
    plt.xticks(fontsize = 20); plt.yticks(fontsize = 20)
    plt.xlim(iter_range)##; plt.ylim(-0.2, 0.2)
    plt.legend(fontsize = 20)
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_PATH, TIME + "_layer-%s.pdf" % l))
    plt.close()
    
    
def difference(L):
    out = [L[0]]
    for i in range(1, len(L)):
        out.append(L[i] - L[i-1])
    return out 
    

def smooth(L, window = 50):
    out = []
    num = len(L)
    for i in range(num):
        if i >= window:
            out.append(np.average(L[i+1-window : i+1])) 
        else:
            out.append(np.average(L[:i]))
    return out
    
    
def plot():
    '''
        Usage: python analyze_weight_log.py  /home/wanghuan/Caffe/Caffe_APP_2/Caffe_APP/MyProject/APP/x00_explore_convnet_cifar10/weights/log_20170920-1958_index.txt
    '''
    assert len(sys.argv) == 2
    global OUTPUT_PATH, LOG_ID, TIME
    OUTPUT_PATH = "/" + os.path.join(*sys.argv[1].split("/")[:-1])
    # LOG_ID = sys.argv[1].split("/")[-3]
    TIME = time.strftime("%Y%m%d-%H%M", time.localtime())
    
    time_stamp = sys.argv[1].split("/")[-1].split("_")[0]
    index  = get_index(OUTPUT_PATH  + "/" + time_stamp + "_log" +  "_index.txt")
    weight = get_weight(OUTPUT_PATH + "/" + time_stamp + "_log" +  "_weight.txt")  
    diff   = get_weight(OUTPUT_PATH + "/" + time_stamp + "_log" +  "_diff.txt")
    
    [plot_sub(i, index[i], weight[i], diff[i]) for i in index]

 
OUTPUT_PATH = ""
LOG_ID = ""
TIME = ""


if __name__ == "__main__":
    plot()
    