from __future__ import print_function
import numpy as np
import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import copy


caffe_root = "/home/wanghuan/Caffe/Caffe_default/"
#caffe_root="/home/wanghuan/Caffe/Caffe_APP/"
sys.path.insert(0, caffe_root + 'python')
import caffe as c

model = "cifar10_full.prototxt"

## Check reconv module 
def check_reconv():
    model = "reconv_deploy.prototxt"
    weights = "_iter_120000.caffemodel.h5"
    net = c.Net(model, weights, c.TEST)
    w2_1 = net.params['conv2_1'][0].data.flatten()
    w2_2 = net.params['conv2_2'][0].data.flatten()
    w3_1 = net.params['conv3_1'][0].data.flatten()
    w3_2 = net.params['conv3_2'][0].data.flatten()
    cos2 = np.dot(w2_1, w2_2) / np.sqrt(np.dot(w2_1, w2_1)) / np.sqrt(np.dot(w2_2, w2_2))
    cos3 = np.dot(w3_1, w3_2) / np.sqrt(np.dot(w3_1, w3_1)) / np.sqrt(np.dot(w3_2, w3_2))
    print (cos2, cos3)


def randomly_prune_col(net, layer, prune_ratio):
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    rands = np.random.permutation(range(w.shape[1]))[:int(prune_ratio * w.shape[1])]

    for i in rands:
        w[:, i] = 0
    w.shape = net.params[layer][0].data.shape
    net.params[layer][0].data[:] = w[:]
    return net

    
## Prune cols
def prune(weights, outpath):
    net_ = c.Net("cifar10_full.prototxt", weights, c.TEST)
    net_ = randomly_prune_col(net_, 'conv2', 0.625)
    net_ = randomly_prune_col(net_, 'conv3', 0.625)
    net_.save(outpath)

# -----------------------------------------------------------  
def prune_layer(net, layer, prune_ratio):
    '''
        Given some layer in net, prune the most unimportant cols, with criteria of col abs ave.
    '''
    print ("current processing: " + layer)
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    
    w_abs = np.abs(w)
    w_colave = np.average(w_abs, axis = 0)
    order = np.argsort(w_colave)[: int(np.ceil(prune_ratio * w.shape[1]))] # in ascending order
    print ("the col to prune:", order[:])
    
    w[:, order] = 0
    w.shape = net.params[layer][0].data.shape
    net.params[layer][0].data[:] = w[:]
    return net
    
  
## prune the most unimportant cols    
def prune_unimportant_cols(weights, prune_ratio, model):
    net = c.Net(model, weights, c.TEST)
    for layer_name, param in net.params.iteritems():
        if len(param[0].data.shape) == 2:
            continue # do not prune fc layers
        net = prune_layer(net, layer_name, prune_ratio)
    net.save(weights.split('.caffemodel')[0] + '_pruned' + '.caffemodel')
# -----------------------------------------------------------

    

# -----------------------------------------------------------
def clear_layer(net, layer, ABS_AVE_THRESHOLD):
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    w_abs = np.abs(w)

    # clear cols
    index_of_pruned_col = np.where(np.average(w_abs, axis = 0) < ABS_AVE_THRESHOLD)[0] 
    w[:, index_of_pruned_col] = 0
    
    # clear rows
    clear_row = True
    if clear_row == True:
        index_of_pruned_row = np.where(np.average(w_abs, axis = 1) < ABS_AVE_THRESHOLD)[0] 
        w[:, index_of_pruned_row] = 0
    
    # save
    w.shape = net.params[layer][0].data.shape
    net.params[layer][0].data[:] = w[:]
    print ("layer {} cleared, ABS_AVE_THRESHOLD = {}, clear_row = {}".format(layer, ABS_AVE_THRESHOLD, clear_row))
    return net

## clear 1e-45 in trained model to avoid their slight influence
def clear_minimals(weights, model, ABS_AVE_THRESHOLD):
    net = c.Net(model, weights, c.TEST)
    for layer_name, param in net.params.iteritems():
        if len(param[0].data.shape) == 2:
            continue # do not clear fc layers
        net = clear_layer(net, layer_name, ABS_AVE_THRESHOLD)
    net.save(weights.split('.caffemodel')[0] + '_cleared' + '.caffemodel')
# -----------------------------------------------------------

## col and row variance
def check_var():
    colvar = {}; rowvar = {}
    model = "cifar10_full.prototxt"; weights = "colregL1_iter_120000.caffemodel.h5"
    net = c.Net(model, weights, c.TEST)
    for layer_name, param in net.params.iteritems():
        w = param[0].data
        if len(w.shape) == 4:
            w.shape = [w.shape[0], -1]
            colvar[layer_name] = [np.var(w[:, j]) for j in range(w.shape[1])]
            rowvar[layer_name] = [np.var(w[i, :]) for i in range(w.shape[0])]        
        
        else:
            continue
        figure = plt.figure()
        plt.subplot(121)
        plt.hist(colvar[layer_name], bins = 'auto')
        plt.title(layer_name + "_colvar")
        
        plt.subplot(122)
        plt.hist(rowvar[layer_name], bins = 'auto')
        plt.title(layer_name + "_rowvar")
        
        figure.savefig(weights.split('_')[0] + layer_name + "_var.jpg")

# ---------------------------------------------------------------------
# Clear zero rows in caffemodel and make new net for it       
def get_num_zero_row(model, weights):
    net = c.Net(model, weights, c.TEST)
    num_zero_row = {}
    for layer, param in net.params.iteritems():
        num_zero_row[layer] = 0
        w = param[0].data
        if len(w.shape) != 4: continue
        num_row = w.shape[0]
        for i in range(num_row):
            if np.sum(np.abs(w[i])) == 0:
                num_zero_row[layer] += 1
        print ("%d rows of layer %s can be pruned." % (num_zero_row[layer], layer))
    return num_zero_row
                
def make_new_net(num_zero_row, model_old):
    lines = [l for l in open(model_old)]
    out_net = open(model_old.replace(".prototxt", "_pruned.prototxt"), "w+")
    for i in range(len(lines)):
        new_l = lines[i]
        if "num_output" in lines[i]:
            # get layer type
            k = 1
            print (lines[i])
            while (not "convolution_param" in lines[i-k]) or (not "inner_product_param" in lines[i-k]):
                k += 1
                print (lines[i-k])
            layer_type = "Conv" if "convolution_param" in lines[i-k] else "FC"
            print (layer_type)
            if layer_type == "Conv":
                # get layer name
                k = 1
                while ~("name" in lines[i-k]):
                    k += 1
                layer_name = lines[i-k].split('"')[1]
                num_row_origin = int(lines[i].split("num_output:")[1].strip())
                print (num_row_origin, num_zero_row[layer_name])
                new_l = lines[i].split("num_output")[0] + "num_output: " + str(num_row_origin - num_zero_row[layer_name])
        out_net.write(new_l)
    out_net.close()
    print ("make new net done.")


def clear_bias(model, weights, model_new):
    net_old = c.Net(model, weights, c.TEST)
    net_new = c.Net(model_new, c.TRAIN)
    for layer, param in net_old.params.iteritems():
        net_new.params[layer][0].data[:] = param[0].data[:]
        if "fc" in layer:
            net_new.params[layer][1].data[:] = param[1].data[:]
    net_new.save(weights.replace(".caffemodel", "_no-bias.caffemodel"))
    
def get_connected_layer(model):
    lines = [l.strip() for l in open(model)]
    result = []
    for i in range(len(lines)):
        if "eltwise_param" in l:
            k = 1
            pair = []
            while not ("name" in lines[i-k]):
                if "bottom" in lines[i-k]:
                    pair.append(lines[i-k].split('"')[1])
                k += 1
            result.append(pair)
    return result
            
def shrink_ind_layer(net_old, net_new, layer):
    pass
def shrink_dep_layer(net_old, net_new, layer1, layer2):
    w1 = net_old.params[layer1]
    w2 = net_old.params[layer2]
    ix1, ix2 = [], []
    w1_tmp, w2_tmp = [], []
    for i in range(w1.shape[0]):
        if np.abs(w1[i]).sum() != 0:
            w1_tmp.append(w1[i])
            ix1.append(i)
    for i in range(w2.shape[0]):
        pass
            
    
    
    
def make_new_caffemodel(model_old, weights_old, model_new):
    assert weights_old.endswith("caffemodel") or weights_old.endswith("caffemodel.h5")
    mode = c.TRAIN if "train" in model_old else c.TEST
    net_old = c.Net(model_old, weights_old, mode)
    net_new = c.Net(model_new, c.TRAIN)
    # pair = get_connected_layer(model_old)
    
    last_layer = ""
    ix_nzero_row = {} ## index of non-zero row
    cnt = 0
    ix_out = []
    for layer, param in net_old.params.iteritems(): 
        print ("%2d: Process layer %s, num param:%s" % (cnt, layer, len(param)))
        cnt += 1
        w = param[0].data
        
        if "relu" in layer:
            w_tmp = w[ix_nzero_row[last_layer]]
        else:
            w_tmp, b_tmp, ix = [], [], []
            num_row = w.shape[0]
            IF_bias = True if len(param) == 2 else False
            b = param[1].data if IF_bias else None
            
            # determine layer is connected by bypass
            # IF_indep = if layer in pair

            # remove zero rows
            for i in range(num_row):
                if np.abs(w[i]).sum() != 0:
                    w_tmp.append(w[i])
                    ix.append(i)
                    if IF_bias:
                        b_tmp.append(b[i])
   
            ix_nzero_row[layer] = ix
            ix_out.append(ix)
            
            w_tmp = np.array(w_tmp)
            b_tmp = np.array(b_tmp)
            
            # reshape fc weights to 4-d like conv weights, for convenience of pruning channel
            if "fc" in layer:
                channel = net_old.params[last_layer][0].data.shape[0]
                w_tmp.shape = (w_tmp.shape[0], channel, 1, -1)
        
            # clear zombie cols
            w_tmp = w_tmp[:, ix_nzero_row[last_layer]] if last_layer else w_tmp
            last_layer = layer
            
            # reshape back
            if "fc" in layer:
                w_tmp = w_tmp.copy()
                w_tmp.shape = (w_tmp.shape[0], -1)

        net_new.params[layer][0].data[:] = w_tmp[:]
        
    net_new.save(weights_old.replace(".caffemodel", "_pruned.caffemodel"))
    np.savetxt("ix_nzero_row.txt", ix_out, fmt = "%s", delimiter = "\n")
        

def clear_zeros(model, weights):
    # num_zero_row = get_num_zero_row(model, weights)
    # make_new_net(num_zero_row, model)
    # clear_bias("face.prototxt", "retrain_iter_38000.caffemodel", "face_no-bias.prototxt")
    make_new_caffemodel("train_val.prototxt", "retrain_iter_53000.caffemodel", "train_val_shrinked.prototxt")
    #net1 = c.Net("face_no-bias.prototxt",        "retrain_iter_38000_no-bias.caffemodel",        c.TEST)
    #net2 = c.Net("face_no-bias_pruned.prototxt", "retrain_iter_38000_no-bias_pruned.caffemodel", c.TEST)
    #compare_net(net1, net2, "conv4")
    #print ("Done.")

def compare_net(net1, net2, output_layer):
    image = "cat.png"
    imdata = c.io.load_image(image)
    result = []
    transformer = c.io.Transformer({'data': net1.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformed_image = transformer.preprocess('data', imdata)
    for net in (net1, net2):
        net.blobs["data"].data[...] = transformed_image
        net.forward()
        result.append(net.blobs[output_layer].data[0])
    difference = np.abs(result[0]).sum() - np.abs(result[1]).sum()
    print ("diff of feature map of layer %s:" % output_layer, difference)
    
    
    # for layer, param in net1.params.iteritems():
        # w1 = param[0].data
        # w2 = net2.params[layer][0].data
        # print ("diff of weight of layer %s:" % layer, np.abs(w1 - w2).sum())
  
    
    return difference
# ---------------------------------------------------------------------
  
def arg_parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', type = str, help = 'weights')
    parser.add_argument('--model', '-m', type = str, help = 'model')
    parser.add_argument('--prune_ratio', '-P', type = float, help = 'prune_ratio', default = 0)
    parser.add_argument('--layer', '-l', type = str, help = 'layer name', default = None)
    parser.add_argument('--ABS_AVE_THRESHOLD', '-A', type = float, help = 'ABS_AVE_THRESHOLD', default = 0)
    return parser.parse_args(args)
    
def main():
 
 

if __name__ == "__main__":
    main()






