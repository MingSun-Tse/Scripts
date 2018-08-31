from __future__ import print_function
import numpy as np
import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import copy
import cPickle as pickle

caffe_root = "/home2/wanghuan/Caffe/Caffe_default/"
#caffe_root="/home/wanghuan/Caffe/Caffe_APP/"
sys.path.insert(0, caffe_root + 'python')
import caffe as c

def get_recover(prune_log):
    col_recovered = {}
    for l in open(prune_log):
        if "recover prob:" in l:
            col = l.split(" ")[2]
            prob = float(l.split(" ")[6])
            if prob < 1:
                if col in col_recovered.keys():
                    col_recovered[col] += 1
                else:
                    col_recovered[col] = 1
    # print (col_recovered)
    f = file("col_recovered.pkl", 'wb')
    pickle.dump(col_recovered, f)
    f.close()

    
def plot():
    pass
    

def main():
    model   = sys.argv[1]
    weights = sys.argv[2]
    FLAG = c.TRAIN if "train" in model else c.TEST
    net = c.Net(model, weights, FLAG)
    
    get_recover(sys.argv[3]) # prune_log
    f = file("col_recovered.pkl", "rb")
    col_recovered = pickle.load(f) 
    

    for layer, param in net.params.iteritems():
        w = param[0].data
        if len(w.shape) != 4: continue
        w.shape = [w.shape[0], -1]
        w_ = np.abs(w)
        w_colave = np.average(w_, axis = 0)
        nonzero_cols = np.where(w_colave != 0)[0]
        print ("\nnum of nonzero cols of %s: %d" % (layer, len(nonzero_cols)))
        
        # how many recovered cols in each layer
        cnt = 0
        for k in col_recovered.keys():
            if layer in k:
                cnt += 1
        print ("num of recovered cols in %s: %d" % (layer, cnt))
        
        # how many survived were once recovered
        cnt = 0
        for col in [layer + "-" + str(i) for i in nonzero_cols]:
            if col in col_recovered:
                print (col+":", col_recovered[col])
                cnt += 1
        print ("%d survived cols in %s were once recovered" % (cnt, layer))
        
        
        
        
if __name__ == "__main__":
    main()
            
        
            