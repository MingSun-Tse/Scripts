from __future__ import print_function

import numpy as np
import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import copy

caffe_root = "/home/wanghuan/Caffe/Caffe_default"
sys.path.insert(0, os.path.join(caffe_root, "python"))
import caffe as c
 
class Layer():
    def __init__(self, name):
        self.name  = name
        self.index = 0
        self.group = 1
        self.prune_ratio = 0
        
def get_net(model):
    lines = [l.strip() for l in open(model, 'r')]
    cnt = 0
    net = {}    
    for i in range(len(lines)):
        if "convolution_param" in lines[i]:
            # get layer name
            k = 1
            while not("name" in lines[i-k]):
                k += 1
            layer_name = lines[i-k].split('"')[1]
            layer = Layer(layer_name)
            
            # get layer index
            layer.index = cnt 
            cnt += 1
            
            # get group
            k = 1
            while not("group" in lines[i+k]) and not("layer" in lines[i+k]):
                k += 1
            layer.group = int(lines[i+k].split(":")[1].strip()) if "group" in lines[i+k] else 1
            net[layer_name] = layer
    return net


def change_batch_size(prototxt):
    lines = [l for l in open(prototxt)]
    os.remove(prototxt)
    with open(prototxt, "w+") as outfile:
        for l in lines:
            new_l = l
            if "batch_size" in l:
                new_l = l.split("batch_size")[0] + "batch_size: 1\n"
        outfile.write(new_l)

def filter_(x):
    x = np.array(x)
    assert (len(x.shape) == 2)
    out = x
    num_row = x.shape[0]
    num_col = x.shape[1]
    for i in range(num_row):
        for j in range(num_col):
            out[i, j] = 1 if x[i, j] else 0
    return out
        
def check_zeros(model, weights, layer, thr = 0):
    flag = c.TRAIN if "train" in model else c.TEST
    net = c.Net(model, weights, flag)
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    w = np.abs(w)
    
    # get the group
    nnet = get_net(model)
    g  = nnet[layer].group 
    
    num_row_per_g = w.shape[0] / g
    w_ = w[0 : num_row_per_g, :]
    for i in range(1, g):
        tmp = w[i * num_row_per_g : (i + 1) * num_row_per_g, :]
        w_  = np.concatenate((w_, tmp), axis = 1)
    w  = filter_(w)
    w_ = filter_(w_)
    # np.savetxt("weight_" + layer, w, fmt = "%d")
    # np.savetxt("weight_reshape_" + layer, w_, fmt = "%d")
    w_colave = np.average(w_, axis = 0)
    w_rowave = np.average(w,  axis = 1)
    
    print ("col ave of layer {}:\n{}".format(layer, w_colave[:]))
    print ("row ave of layer {}:\n{}".format(layer, w_rowave[:]))
    
    print ("Group of weight in this layer: %s" % g)
    print ("the cols whose abs ave <= {}:\n{}".format(thr, np.where(w_colave <= thr)[0]))
    print ("the rows whose abs ave <= {}:\n{}".format(thr, np.where(w_rowave <= thr)[0]))
    
    print ("total num of col whose abs ave <= %s: %.2f / %d" % (str(thr), np.sum(w_colave <= thr)*1.0/g, w.shape[1]))
    print ("total num of row whose abs ave <= %s: %d / %d" %   (str(thr), np.sum(w_rowave <= thr)      , w.shape[0]))
    
    return np.where(w_colave <= thr)[0]
    
def arg_parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', type = str, help = 'weights')
    parser.add_argument('--model', '-m', type = str, help = 'model')
    parser.add_argument('--prune_ratio', '-p', type = float, help = 'prune_ratio', default = 0)
    parser.add_argument('--layer', '-l', type = str, help = 'layer_name', default = None)
    parser.add_argument('--ABS_AVE_THRESHOLD', '-A', type = float, help = 'ABS_AVE_THRESHOLD', default = 0)
    return parser.parse_args(args)
    
if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    check_zeros(args.model, args.weights, args.layer, args.ABS_AVE_THRESHOLD)
    