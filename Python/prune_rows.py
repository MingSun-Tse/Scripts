from __future__ import print_function
import os
import sys
import numpy as np
caffe_root = "/home/wanghuan/Caffe/Caffe_default"
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe as c
import pickle 
'''
This file is to prune corresponding rows from a *column-pruned* caffemodel, including Conv layers and FC layers.

'''

def Slim(model, caffemodel, model_slimmed):
    FLAG = c.TRAIN if "train" in model else c.TEST
    net  = c.Net(model, caffemodel, FLAG)
    net_slimmed = c.Net(model_slimmed, FLAG)
    layers  = []
    weights = []
    biases  = []
    pruned_rows = {}
    pruned_cols = {}
    saved_rows  = {}
    saved_cols  = {}
    num_conv = 0
    
    # Prune columns
    for layer, param in net.params.iteritems():
        print("dealing with %s" % layer)
        w = param[0].data
        if len(w.shape) not in [2, 4]: 
            continue
        b = param[1].data if len(param) > 1 else [] # TODO(Ming): check bias, and consider BN layer
        if len(w.shape) == 4:
            num_conv += 1
        layers.append(layer)
        weights.append(w[:])
        biases.append(b[:])
        pruned_rows[layer] = []
        saved_rows[layer]  = []
        
        # count pruned cols
        w.shape = [w.shape[0], -1]
        col_sum_abs = np.sum(np.abs(w), axis = 0)
        pruned_cols[layer] = list(np.where(col_sum_abs == 0)[0])
        saved_cols[layer]  = list(np.where(col_sum_abs != 0)[0])
        if len(param[0].data.shape) == 2 and len(layers)-1 != num_conv: # for FC layers but the first FC layer, remove the zero columns
            weights[-1] = weights[-1][:, saved_cols[layer]]
        print ("{}: {} columns ({}) are pruned".format(layer, 
                                                       len(pruned_cols[layer]),
                                                       len(pruned_cols[layer]) * 1.0 / w.shape[1]))
        print (pruned_cols[layer])
    
    # Prune rows
    print("\n")
    for l in range(len(layers)):
        layer_name = layers[l]
        print("dealing with %s" % layer_name)
        num_row = weights[l].shape[0]
        if len(weights[l].shape) == 4: # For Conv layers, directly copy their weights into new model.
            num_param = len(net_slimmed.params[layer_name])
            for i in range(num_param):
                net_slimmed.params[layer_name][i].data[:] = net.params[layer_name][i].data[:]
        # TODO(Ming): slim Conv layers
        else:
            if l == len(layers) - 1: # the last FC layer
                net_slimmed.params[layer_name][0].data[:] = weights[l][:]
                net_slimmed.params[layer_name][1].data[:] =  biases[l][:]
            else:
                for r in range(num_row):
                    if r in pruned_cols[layers[l+1]]:
                        pruned_rows[layer_name].append(r)
                    else:
                        saved_rows[layer_name].append(r)
                net_slimmed.params[layer_name][0].data[:] = weights[l][saved_rows[layer_name]]
                net_slimmed.params[layer_name][1].data[:] =  biases[l][saved_rows[layer_name]]

        #net.params[layer_name][0].data[:] = weights[l][:]
        print ("{}: {} rows of {} can be removed safely:".format(layer_name, 
                                                                   len(pruned_rows[layer_name]),
                                                                   num_row))
    #net.save(weights.split(".caffemodel")[0] + "_row-zeroed.caffemodel")
    net_slimmed.save(caffemodel.split(".caffemodel")[0] + "_row-removed.caffemodel")

# IF_col_continously_pruned = True
# for j in range(r * filter_area_next_layer, (r + 1) * filter_area_next_layer):
    # if sum(np.abs(weights_next_layer[:, j])) != 0:
        # IF_col_continously_pruned = False
        # break
    
if __name__ == "__main__":
    '''Usage:
        python  this_file.py  train_val/deploy.prototxt  xxx.caffemodel  train_val_slimmed.prototxt
    '''
    assert(len(sys.argv) == 4)
    Slim(*sys.argv[1:])
