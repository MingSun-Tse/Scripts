from __future__ import print_function
import os
import sys
import numpy as np
caffe_root = "/home2/wanghuan/Caffe/Caffe_default"
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe as c
import pickle

def remove_continuous_zero_columns(weights, filter_spatial):
  original_shape = weights.shape
  weights.shape = weights.shape[0], -1 # expanded into matrix
  num_row, num_col = weights.shape
  saved_cols = []
  saved_rows_last_layer  = []
  pruned_rows_last_layer = []
  assert(num_col % filter_spatial == 0)
  num_row_last_layer = original_shape[1] # i.e. num_channel
  for r in range(num_row_last_layer):
    if np.sum(np.abs(weights[:, r * filter_spatial : (r+1) * filter_spatial])) == 0:
      pruned_rows_last_layer.append(r)
    else:
      saved_rows_last_layer.append(r)
      saved_cols += range(r * filter_spatial, (r+1) * filter_spatial)
  
  weights.shape = original_shape
  weights = weights[:, saved_rows_last_layer, :, :] # save the unpruned columns
  return weights, saved_rows_last_layer, pruned_rows_last_layer
  
def Slim(model, caffemodel, model_slimmed = None):
    FLAG = c.TRAIN if "train" in model else c.TEST
    net  = c.Net(model, caffemodel, FLAG)
    
    layers, weights, biases  = [], [], []
    pruned_rows, pruned_cols = {}, {}
    saved_rows,   saved_cols = {}, {}
    filter_spatial = {}
    num_conv, num_fc = 0, 0
    # Prune columns, just based on existing masks
    for layer, param in net.params.iteritems():
        print("\n*** dealing with %s ***" % layer)
        w = param[0].data
        if len(w.shape) not in [2, 4]: 
            continue
        b = param[1].data if len(param) > 1 else [] # TODO(Ming): check bias, and consider BN layer
        if len(w.shape) == 4:
            num_conv += 1
            filter_spatial[layer] = w.shape[2] * w.shape[3]
        else: 
            num_fc += 1
        layers.append(layer)
        weights.append(w[:])
        biases.append(b[:])
        
        # count pruned cols
        w.shape = w.shape[0], -1
        num_row, num_col = w.shape
        col_sum_abs = np.sum(np.abs(w), axis = 0)
        row_sum_abs = np.sum(np.abs(w), axis = 1)
        pruned_cols[layer] = list(np.where(col_sum_abs == 0)[0])
        saved_cols[layer]  = list(np.where(col_sum_abs != 0)[0])
        print ("{}: {} columns of {} ({}) are pruned".format(layer, len(pruned_cols[layer]), num_col, len(pruned_cols[layer]) * 1.0 / num_col))
        print (pruned_cols[layer])
        
        pruned_rows[layer] = list(np.where(row_sum_abs == 0)[0])
        saved_rows[layer]  = list(np.where(row_sum_abs != 0)[0])
        print ("{}: {} rows of {} ({}) are pruned (before analysis based on row-col dependency)".format(layer, 
                  len(pruned_rows[layer]), num_row, len(pruned_rows[layer]) * 1.0 / num_row))
        print (pruned_rows[layer])
        
        # Remove columns if they are continuously pruned
        if len(param[0].data.shape) == 2 and num_fc != 1: # Not the first FC
          weights[-1] = weights[-1][:, saved_cols[layer]]
          weights[-2] = weights[-2][saved_cols[layer], :]
          biases[-2]  =  biases[-2][saved_cols[layer]]
        if len(param[0].data.shape) == 4 and num_conv != 1: # Not the first Conv
          weights[-1], saved_rows_last_layer, pruned_rows_last_layer = remove_continuous_zero_columns(weights[-1], filter_spatial[layer])
          pruned_rows[layers[-2]] = pruned_rows_last_layer
          print ("{}: {} rows of {} ({}) can be removed safely (after analysis based on row-col dependency)".format(layers[-2], 
                len(pruned_rows[layers[-2]]), weights[-2].shape[0], len(pruned_rows[layers[-2]]) * 1.0 / weights[-2].shape[0]))
          print (pruned_rows[layers[-2]])
          weights[-2] = weights[-2][saved_rows_last_layer, :]
          biases[-2]  =  biases[-2][saved_rows_last_layer]
          
    # Save new caffemodels
    print("========> Zero the pruned rows")
    for layer, param in net.params.iteritems():
      w = param[0].data
      if len(w.shape) != 4: continue # only consider Conv
      num_col = w.shape[1]* w.shape[2]* w.shape[3]
      for pruned_row in pruned_rows[layer]:
        net.params[layer][0].data[pruned_row][:] = np.zeros(w.shape[1:])
    net.save(caffemodel.split(".caffemodel")[0] + "_row-zeroed.caffemodel")
    if model_slimmed:
      print("========> Remove the pruned_rows")
      net_slimmed = c.Net(model_slimmed, FLAG)
      for l in range(len(layers)):
        layer_name = layers[l]
        net_slimmed.params[layer_name][0].data[:] = weights[l]
        if len(net_slimmed.params[layer_name]) == 2:
          net_slimmed.params[layer_name][1].data[:] = biases[l]
      net_slimmed.save(caffemodel.split(".caffemodel")[0] + "_row-removed.caffemodel")
    
if __name__ == "__main__":
    '''
      This file is to prune corresponding rows from a *column-pruned* caffemodel, including Conv layers and FC layers.
      Usage:
        1. python  this_file.py  train_val/deploy.prototxt  xxx.caffemodel  # Only check pruned rows and columns
        2. python  this_file.py  train_val/deploy.prototxt  xxx.caffemodel  train_val_slimmed.prototxt # Check pruned rows and columns, and remove them
      Output:
        1. row-zeroed.caffemodel
        2. row-zeroed.caffemodel (shape not changed), row-removed.caffemodel (shape changed)
    '''
    assert(len(sys.argv) in [3, 4])
    Slim(*sys.argv[1:])
