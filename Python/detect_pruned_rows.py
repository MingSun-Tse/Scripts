from __future__ import print_function
import os
import sys
import numpy as np
caffe_root = "/home2/wanghuan/Caffe/Caffe_default"
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe as c
import pickle
  
def Slim(model, caffemodel, model_slimmed = None):
    FLAG = c.TRAIN if "train" in model else c.TEST
    net  = c.Net(model, caffemodel, FLAG)
    pruned_cols = {}
    num_conv = 0

    for layer, param in net.params.iteritems():
        w = param[0].data
        if len(w.shape) != 4 : continue
        num_conv += 1
        num_row = w.shape[0]
        num_chl = w.shape[1]
        height = w.shape[2]
        width = w.shape[3]
        num_col = num_chl * width * height
        
        # count pruned cols
        w.shape = w.shape[0], -1
        col_sum_abs = np.sum(np.abs(w), axis = 0)
        pruned_cols[layer] = list(np.where(col_sum_abs == 0)[0])
        print ("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (num_conv,
                                                       layer, 
                                                       num_row, num_chl, height, width,
                                                       num_col,
                                                       len(pruned_cols[layer]),
                                                       len(pruned_cols[layer]) * 1.0/ num_col))
    
if __name__ == "__main__":
    '''
      This file is to prune corresponding rows from a *column-pruned* caffemodel, including Conv layers and FC layers.
      Usage:
        python  this_file  deploy.prototxt  xxx.caffemodel  # Only check pruned rows and columns
    '''
    assert(len(sys.argv) in [3, 4])
    Slim(*sys.argv[1:])
