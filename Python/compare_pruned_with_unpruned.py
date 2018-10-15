from __future__ import print_function
import os
import sys
import numpy as np
caffe_root = "/home/wanghuan/Caffe/Caffe_default"
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe as c
import pickle 
'''

'''  
def Compare(model, basemodel, prunedmodel, finalmodel):
    FLAG = c.TRAIN if "train" in model else c.TEST
    basenet   = c.Net(model, basemodel, FLAG)
    prunednet = c.Net(model, prunedmodel, FLAG)
    finalnet  = c.Net(model, finalmodel, FLAG)
    
    saved_cols = {}
    saved_rows = {}
    # Prune columns
    for layer, _ in basenet.params.iteritems():
        baseweight   = basenet.params[layer][0].data
        prunedweight = prunednet.params[layer][0].data
        finalweight  = finalnet.params[layer][0].data
        if len(baseweight.shape) != 4: continue
        print("\ndealing with layer %s:" % layer)
        
        baseweight.shape   = baseweight.shape[0], -1
        prunedweight.shape = prunedweight.shape[0], -1
        finalweight.shape  = finalweight.shape[0], -1
        
        prunedweight_col_abs = np.sum(np.abs(prunedweight), axis = 0)
        prunedweight_row_abs = np.sum(np.abs(prunedweight), axis = 1)
        finalweight_col_abs  = np.sum(np.abs(finalweight), axis = 0)
        finalweight_row_abs  = np.sum(np.abs(finalweight), axis = 1)
        
        assert(list(np.where(finalweight_col_abs != 0)[0]) == list(np.where(prunedweight_col_abs != 0)[0]))
        assert(list(np.where(finalweight_row_abs != 0)[0]) == list(np.where(prunedweight_row_abs != 0)[0]))
        saved_cols[layer] = list(np.where(finalweight_col_abs != 0)[0])
        saved_rows[layer] = list(np.where(finalweight_row_abs != 0)[0])
        print("num_pruned_col = {}, num_pruned_row = {}".format(baseweight.shape[1] - len(saved_cols[layer]),
                                                                baseweight.shape[0] - len(saved_rows[layer])))

        # Print
        baseweight_vec   =   baseweight[saved_rows[layer], :][:, saved_cols[layer]].flatten()
        prunedweight_vec = prunedweight[saved_rows[layer], :][:, saved_cols[layer]].flatten()
        finalweight_vec  =  finalweight[saved_rows[layer], :][:, saved_cols[layer]].flatten()
        diff_final_pruned = np.linalg.norm(finalweight_vec - prunedweight_vec, 2)
        diff_final_base   = np.linalg.norm(finalweight_vec - baseweight_vec, 2)
        print("the magnitude of base   weights = {}".format(np.linalg.norm(baseweight_vec, 2)))
        print("the magnitude of pruned weights = {}".format(np.linalg.norm(prunedweight_vec, 2)))
        print("the magnitude of final  weights = {}".format(np.linalg.norm(finalweight_vec, 2)))
        print("the difference between final weights and pruned weights = {}\nthe difference between final weights and   base weights = {}".format(diff_final_pruned, diff_final_base))
        
    
if __name__ == "__main__":
    '''Usage:
        python  this_file.py  train_val/deploy.prototxt  baseline.caffemodel  pruned.caffemodel  final.caffemodel
    '''
    assert(len(sys.argv) == 5)
    Compare(*sys.argv[1:])
