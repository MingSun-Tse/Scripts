from __future__ import print_function
import os
import sys
import numpy as np
caffe_root = "/home/wanghuan/Caffe/Caffe_default"
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe as c
import pickle 
        
def prune_rows(model, weights):
    FLAG = c.TRAIN if "train" in model else c.TEST
    net2 = c.Net(model, weights, FLAG) # row_pruned model, output
    layers = []
    ws = []
    pruned_rows = {}
    pruned_cols = {}
    
    for layer, param in net2.params.iteritems():
        w = param[0].data
        if len(w.shape) != 4: continue
        layers.append(layer)
        ws.append(w[:])
        pruned_rows[layer] = []
        
        # count pruned cols
        w.shape = [w.shape[0], -1]
        w_abs = np.abs(w)
        colsum_abs = np.sum(w_abs, axis = 0)
        pruned_cols[layer] = list(np.where(colsum_abs == 0)[0])
        print ("{}: {} columns ({}) are pruned".format(layer, len(pruned_cols[layer]), len(pruned_cols[layer])*1.0/w.shape[1]))
        np.savetxt(weights.replace(".caffemodel", "_pruned_cols_%s.txt" % layer), pruned_cols[layer], fmt = "%d", delimiter = " ")
    with open(weights.replace(".caffemodel", "_pruned_cols.pkl"), 'w') as f:
        pickle.dump(pruned_cols, f)
        
    
    for l in range(len(layers) - 1): # l: the layer index of conv layers
        layer = layers[l]
        num_row = ws[l].shape[0]
        filter_area_next_layer = ws[l+1].shape[2] * ws[l+1].shape[3]
        ws_next_layer = ws[l+1][:]
        ws_next_layer.shape = [ws_next_layer.shape[0], -1]
        for i in range(num_row):
            ii = i % num_row # TODO: add group, i % (num_row / group_[layers[l+1]])
            IF_col_continously_pruned = True
            for j in range(ii * filter_area_next_layer, (ii + 1) * filter_area_next_layer):
                if sum(np.abs(ws_next_layer[:, j])) != 0:
                    IF_col_continously_pruned = False
                    break
            if IF_col_continously_pruned:
                ws[l][i] = np.zeros(ws[l].shape[1:])
                pruned_rows[layers[l]].append(i)
        net2.params[layers[l]][0].data[:] = ws[l][:]
        print ("{}: {} rows can be removed safely: \n{}\n".format(layer, len(pruned_rows[layer]), pruned_rows[layer]))
    net2.save(weights.split(".caffemodel")[0] + "_rowpruned.caffemodel")
    return pruned_rows

    
if __name__ == "__main__":
    prune_rows(*sys.argv[1:])
