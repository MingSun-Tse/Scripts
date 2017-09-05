import sys
import os
import numpy as np
import pickle

def rename_layer(model_old, weights_old, model_new, IF_save):
    '''
        IF_save: save the new layer names and weights into .npy file
        Usage: python convert_caffemodel.py  model_old  weights_old  model_new  0/1
    '''
    CAFFE_ROOT = "/home/wanghuan/Caffe/Caffe_default"
    sys.path.insert(0, os.path.join(CAFFE_ROOT, "python"))
    import caffe as c
    
    net_old = c.Net(model_old, weights_old, c.TEST)
    net_new = c.Net(model_new, weights_old, c.TEST)
    weights = {}
    
    for layer, param in net_new.params.iteritems():
        layer_old = layer
        if layer[0] == "[" and layer.split("[")[1].split("]")[0].isdigit():
            layer_old = layer.split("]")[1]
            
        for k in range(len(param)):
            net_new.params[layer][k].data[:] = net_old.params[layer_old][k].data[:] 
            if layer in weights.keys():
                weights[layer].append(net_old.params[layer_old][k].data)
            else:
                weights[layer] = [net_old.params[layer_old][k].data]
        
    net_new.save(weights_old.split(".caffemodel")[0] + "_renamed.caffemodel")
    if IF_save:
        save_name = weights_old.split(".caffemodel")[0] + "_renamed_weights.pickle"
        with open(save_name, 'wb') as f:
            pickle.dump(weights, f)
        
        ## np.save(weights_old.split(".caffemodel")[0] + "_renamed_weights.npy", weights) 
        ## why there lies problem when using .npy to save dict? 

def rename_layer_add_param(weights_old, model_new, weights_new):
    '''
        Usage: python convert_caffemodel.py  weights_old  model_new  weights_new
    '''
    assert weights_old.split(".")[-1] == "pickle"
    CAFFE_ROOT = "/home/wanghuan/Caffe/Caffe_pruning_filter"
    sys.path.insert(0, os.path.join(CAFFE_ROOT, "python"))
    import caffe as c
    
    net_new = c.Net(model_new, weights_new, c.TEST)
    with open(weights_old, "rb") as f:
        weights = pickle.load(f)
    for layer, param in net_new.params.iteritems():
        for k in range(len(param)):
            net_new.params[layer][k].data[:] = weights[layer][k][:]
    net_new.save(weights_old.split("_weights.pickle")[0] + "_added.caffemodel")

    
if __name__ == "__main__":
    assert len(sys.argv) in (4, 5)
    if len(sys.argv) == 4:
        rename_layer_add_param(*sys.argv[1:])
    elif len(sys.argv) == 5:
        rename_layer(*sys.argv[1:])
        
