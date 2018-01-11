from __future__ import print_function

import numpy as np
import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import copy

caffe_root = "/home/wanghuan/Caffe/Caffe_default"
sys.path.insert(0, os.path.join(caffe_root, "python"))
import caffe as c
from test_acc import Tester

    
def layer_wise_prune(model, weights, prune_ratio):
    FLAG = c.TRAIN if "train" in model else c.TEST
    net  = c.Net(model, weights, FLAG)
    for layer, param in net.params.iteritems():
        w = param[0].data
        if len(w.shape) == 4:
            # net_out = c.Net(model, weights, FLAG)
            w.shape = [w.shape[0], -1]
            np.save("weights_%s.npy" % layer, w) # save layer weights for PCA
            w_abscolsum = np.sum(np.abs(w), axis = 0)
            num_col_to_prune = int(np.ceil(prune_ratio * w.shape[1]))
            ixs = np.argsort(w_abscolsum)[:num_col_to_prune]
            w[:, ixs] = 0
            w.shape = net.params[layer][0].data.shape
           # net_out.params[layer][0].data[:] = w[:]
           # net_out.save(weights.replace(".caffemodel", "_%s_pruned_%s.caffemodel" % (layer, prune_ratio)))
            print ("%s prune finished" % layer)
    print ("all prune finished")

if __name__ == "__main__":
    '''
        python layer_sensitivity_analysis.py  weights_dir/VGG16_train_val.prototxt  weights_dir/VGG16.caffemodel  0.5  50000  4
        Note: in `VGG16_train_val.prototxt`, train batch size should be 1, and val batch size be normal.
    '''
    model, weights, prune_ratio, num_test_examples, gpu = sys.argv[1:]
    layer_wise_prune(model, weights, float(prune_ratio))
    # weights_dir = os.path.join(weights.split("/")[:-1])
    # tester = Tester(model, weights_dir, int(num_test_examples), gpu)
    # tester.test()