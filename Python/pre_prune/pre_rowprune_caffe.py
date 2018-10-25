import sys
import numpy as np
sys.path.insert(0, "/home2/wanghuan/Caffe/Caffe_default/python")
import caffe as c

def filter_prune(weight, num_keeped_row):
  weight.shape = weight.shape[0], -1
  abs_row = np.abs(weight).sum(1)
  order = np.argsort(abs_row)[-num_keeped_row:]
  return order
  
## squeeze the zeroed model into smaller model
def squeeze(model1, weights1, model2):
  net1 = c.Net(model1, weights1, c.TEST)
  net2 = c.Net(model2, c.TRAIN) # the small net
  keeped_row_index = {}
  layers = []
  for layer, param in net2.params.iteritems():
    w1 = net1.params[layer][0].data
    b1 = net1.params[layer][1].data
    w2 = param[0].data
    b2 = param[1].data
    print("processing layer '%s': big model's shape = %s vs. small model's shape = %s" % (layer, w1.shape, w2.shape))
    last_layer_name = layers[-1] if len(layers) else "None"
    layers.append(layer)
    
    # FC layers, not pruned
    if len(w1.shape) == 2:
      net2.params[layer][0].data[:] = w1[:]
      net2.params[layer][1].data[:] = b1[:]
      continue
    
    # get keeped_row_index
    if w1.shape[0] != w2.shape[0]: # don't have the same number of filters
      keeped_row_index[layer] = filter_prune(w1, w2.shape[0])
    else:
      keeped_row_index[layer] = range(w1.shape[0])
    
    # squeeze
    net2.params[layer][1].data[:] = net1.params[layer][1].data[keeped_row_index[layer]] # biases
    if last_layer_name != "None":
      net2.params[layer][0].data[:] = net1.params[layer][0].data[keeped_row_index[layer]][:, keeped_row_index[last_layer_name], :, :]
    else: # the first conv layer
      net2.params[layer][0].data[:] = net1.params[layer][0].data[keeped_row_index[layer]]
  net2.save("slimmed.caffemodel")

if __name__ == "__main__":
    assert(len(sys.argv) == 4)
    model1       = sys.argv[1]
    weights1     = sys.argv[2]
    model2       = sys.argv[3]
    squeeze(model1, weights1, model2)
