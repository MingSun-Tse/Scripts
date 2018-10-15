import sys
import numpy as np
sys.path.insert(0, "/home2/wanghuan/Caffe/Caffe_default/python")
import caffe as c

def prune_layer_col(net, layer, prune_ratio):
    '''
        Given some layer in net, prune the most unimportant cols, with criteria of col abs ave.
    '''
    print ("current processing: " + layer)
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    num_col = w.shape[1]

    w_abs = np.abs(w)
    w_colave = np.average(w_abs, axis = 0) # column
    index = np.argsort(w_colave)[: int(np.ceil(float(prune_ratio) * num_col))] # in ascending order
    print ("the col to prune:", index[:])
    
    w[:, index] = 0
    w.shape = net.params[layer][0].data.shape
    net.params[layer][0].data[:] = w[:]
    return net
    
def prune_layer_row(net, layer, prune_ratio):
    '''
        Given some layer in net, prune the most unimportant cols, with criteria of row abs ave.
    '''
    print ("current processing: " + layer)
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    num_row = w.shape[0]

    w_abs = np.abs(w)
    w_rowave = np.average(w_abs, axis = 1) # row
    index = np.argsort(w_rowave)[: int(np.ceil(float(prune_ratio) * num_row))] # in ascending order
    print ("the row to prune:", index[:])
    
    w[index, :] = 0
    w.shape = net.params[layer][0].data.shape
    net.params[layer][0].data[:] = w[:]
    return net
    
## prune the most unimportant cols
def prune_unimportant_cols(model, weights, prune_ratio):
    net = c.Net(model, weights, c.TEST)
    for layer_name, param in net.params.iteritems():
        if len(param[0].data.shape) != 4:
            continue # do not prune fc layers
        net = prune_layer_col(net, layer_name, prune_ratio)
    net.save(weights.split('.caffemodel')[0] + '_col_pruned' + '.caffemodel')

## prune the most unimportant row
def prune_unimportant_rows(model, weights, prune_ratio):
    net = c.Net(model, weights, c.TEST)
    for layer_name, param in net.params.iteritems():
        if len(param[0].data.shape) != 4:
            continue # do not prune fc layers
        net = prune_layer_row(net, layer_name, prune_ratio)
    net.save(weights.split('.caffemodel')[0] + '_row_pruned' + '.caffemodel')
    
if __name__ == "__main__":
    assert(len(sys.argv) == 5)
    model       = sys.argv[1]
    weights     = sys.argv[2]
    prune_ratio = sys.argv[3]
    mode = sys.argv[4] # "row" or "col"
    if mode == "col":
      prune_unimportant_cols(model, weights, prune_ratio)
    elif mode == "row":
      prune_unimportant_rows(model, weights, prune_ratio)
    else:
      print("wrong")
      exit(1)