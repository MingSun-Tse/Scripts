from __future__ import print_function
import numpy as np
import sys
import os
pjoin = os.path.join
HOME = os.environ["HOME"]
CAFFE_ROOT = pjoin(HOME, "Caffe/Caffe_default")
print(CAFFE_ROOT)
sys.path.insert(0, pjoin(CAFFE_ROOT, "python"))
import caffe

def get_pruned_weight_group(layer_weight):
  layer_weight.shape = [layer_weight.shape[0], -1]
  abs_row = np.sum(np.abs(layer_weight), axis = 1)
  abs_col = np.sum(np.abs(layer_weight), axis = 0)
  return set(np.where(abs_row == 0)[0]), set(np.where(abs_col == 0)[0])
  
def compare(model, weights1, weights2):
  net1 = caffe.Net(model, weights1, caffe.TEST)
  net2 = caffe.Net(model, weights2, caffe.TEST)
  for layer, param in net1.params.iteritems():
    w1 = param[0].data
    w2 = net2.params[layer][0].data
    if len(w1.shape) != 4: continue
    w1.shape = [w1.shape[0], -1]
    num_row, num_col = w1.shape
    zero_row_set1, zero_col_set1 = get_pruned_weight_group(w1)
    zero_row_set2, zero_col_set2 = get_pruned_weight_group(w2)
    row_inter  = zero_row_set1.intersection(zero_row_set2)
    row_diff12 = zero_row_set1.difference(zero_row_set2)
    row_diff21 = zero_row_set2.difference(zero_row_set1)
    col_inter  = zero_col_set1.intersection(zero_col_set2)
    col_diff12 = zero_col_set1.difference(zero_col_set2)
    col_diff21 = zero_col_set2.difference(zero_col_set1)
    
    print("\n%s:" % layer)
    # print("  %s pruned rows (%f) in common: %s" % (len(row_inter), len(row_inter) * 1.0 / num_row, row_inter))
    print("  %s pruned cols (%f) in common: %s" % (len(col_inter), len(col_inter) * 1.0 / num_col, col_inter))
    print("  %s pruned cols (%f) in difference(1-2): %s" % (len(col_diff12), len(col_diff12) * 1.0 / num_col, col_diff12))

if __name__ == "__main__":
  assert(len(sys.argv) == 4)
  model, weights1, weights2 = sys.argv[1:]
  for line in model:
    if "prune_param" in line:
      global CAFFE_ROOT
      CAFFE_ROOT = pjoin(HOME, "Caffe/Caffe_Compression")
      break
  sys.path.insert(0, pjoin(CAFFE_ROOT, "python"))
  import caffe
  compare(model, weights1, weights2)
