from __future__ import print_function
import sys
import util
import numpy as np
import os
CAFFE_ROOT = os.path.join(os.environ["HOME"], "Caffe/Caffe_default")
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe as c

def prune_layer(net, layer, prune_ratio):
    '''
        Given some layer in net, prune the most unimportant cols, with criteria of col abs ave.
    '''
    print ("\nCurrent processing: " + layer)
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    num_col = w.shape[1]

    w_abs = np.abs(w)
    w_colave = np.average(w_abs, axis = 0)
    index = np.argsort(w_colave)[: int(np.ceil(float(prune_ratio) * num_col))] # in ascending order
    print ("the col to prune:\n%s" % set(index[:]))
    
    w[:, index] = 0
    w.shape = net.params[layer][0].data.shape
    net.params[layer][0].data[:] = w[:]
    return net, set(index[:])
    
  
## prune the most unimportant cols
def prune_unimportant_cols(model, weights, prune_ratio):
    net = c.Net(model, weights, c.TEST)
    for layer_name, param in net.params.iteritems():
        if len(param[0].data.shape) != 4 or layer_name not in prune_ratio.keys():
            continue # do not prune fc layers
        net, _ = prune_layer(net, layer_name, prune_ratio[layer_name])
    net.save(weights.split('.caffemodel')[0] + '_pruned' + '.caffemodel')

if __name__ == "__main__":
    assert(len(sys.argv) == 4)
    model, weights, pr_str = sys.argv[1:] # pr example: conv1:0.5-conv2:0.23-conv3:0.6, i.e. layer_name:pr
    prune_ratio = {}
    for i in pr_str.split("-"):
      layer_name = i.split(":")[0].strip()
      pr = float(i.split(":")[1].strip())
      prune_ratio[layer_name] = pr
    prune_unimportant_cols(model, weights, prune_ratio)
