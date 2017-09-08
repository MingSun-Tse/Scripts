import numpy as np
import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse

#caffe_root = "/home/wanghuan/Caffe/Caffe_default/"
caffe_root="/home/wanghuan/Caffe/Caffe_APP/"
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
    print cos2, cos3


def randomly_prune_column(net, layer, PruneRate):
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    rands = np.random.permutation(range(w.shape[1]))[:int(PruneRate * w.shape[1])]

    for i in rands:
        w[:, i] = 0
    w.shape = net.params[layer][0].data.shape
    net.params[layer][0].data[:] = w[:]
    return net

    
## Prune columns
def prune(weights, outpath):
    net_ = c.Net("cifar10_full.prototxt", weights, c.TEST)
    net_ = randomly_prune_column(net_, 'conv2', 0.625)
    net_ = randomly_prune_column(net_, 'conv3', 0.625)
    net_.save(outpath)

# -----------------------------------------------------------  
def prune_layer(net, layer, PruneRate):
    '''
        Given some layer in net, prune the most unimportant columns, with criteria of column abs ave.
    '''
    print "current processing: " + layer
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    
    w_abs = np.abs(w)
    w_colave = np.average(w_abs, axis = 0)
    order = np.argsort(w_colave)[: int(np.ceil(PruneRate * w.shape[1]))] # in ascending order
    print "the column to prune:", order[:]
    
    w[:, order] = 0
    w.shape = net.params[layer][0].data.shape
    net.params[layer][0].data[:] = w[:]
    return net
    
  
## prune the most unimportant columns    
def prune_unimportant_columns(weights, PruneRate, model):
    net = c.Net(model, weights, c.TEST)
    for layer_name, param in net.params.iteritems():
        if len(param[0].data.shape) == 2:
            continue # do not prune fc layers
        net = prune_layer(net, layer_name, PruneRate)
    net.save(weights.split('.caffemodel')[0] + '_pruned' + '.caffemodel')
# -----------------------------------------------------------      

def get_coincidence(array1, array2, array3):
    cnt = 0
    for i in range(len(array1)):
        if array1[i] in array2 and array1[i] in array3:
            cnt += 1
    return (cnt, 
            float(cnt) / len(array1), 
            float(cnt) / len(array2), 
            float(cnt) / len(array3))


    
def check_minimals(weights, layer, ABS_AVE_THRESHOLD, model):
    net = c.Net(model, weights, c.TEST)
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    w = np.abs(w) # use abs to detect small columns
    w_colave = np.average(w, axis = 0)
    w_rowave = np.average(w, axis = 1)

    #w_colmax = np.max(w, axis = 0)
    #w_colmin = np.min(w, axis = 0)
    print "col ave of layer %s:" % layer
    print w_colave[:]    
    
    print "row ave of layer %s:" % layer
    print w_rowave[:]
     
    print "the columns whose abs ave < %s:" % str(ABS_AVE_THRESHOLD)
    print np.where(w_colave < ABS_AVE_THRESHOLD)[0]
    np.save(weights.split(".caffemodel")[0] + "_" + layer + "_pruned_cols.npy", np.where(w_colave < ABS_AVE_THRESHOLD)[0])
    
    print "the rows whose abs ave < %s:" % str(ABS_AVE_THRESHOLD)
    print np.where(w_rowave < ABS_AVE_THRESHOLD)[0]
    np.save(weights.split(".caffemodel")[0] + "_" + layer + "_pruned_rows.npy", np.where(w_rowave < ABS_AVE_THRESHOLD)[0])

    print "total num of column whose abs ave < %s: %d / %d" % (str(ABS_AVE_THRESHOLD), np.sum(w_colave < ABS_AVE_THRESHOLD), w.shape[1])
    print "total num of row whose abs ave < %s: %d / %d" % (str(ABS_AVE_THRESHOLD), np.sum(w_rowave < ABS_AVE_THRESHOLD), w.shape[0])
    
    # plot histgram
    # figure = plt.figure()
    # plt.hist(w_colave, bins = 'auto')
    # plt.title(layer + "_colave")        
    # figure.savefig(weights.split('.caffemodel')[0] + "_" + layer_name + "_colave.jpg")
    
    return np.where(w_colave < ABS_AVE_THRESHOLD)[0]


def check_coincidence(model, weights1, weights2, weights3, layer):
    ''' 
        As a method to select columns to prune, how robust is SSL?
        Does it always choose roughly the same columns in different run?
    '''
    array1 = check_zero(model, weights1, layer)
    array2 = check_zero(model, weights2, layer)
    array3 = check_zero(model, weights3, layer)
    
    co_num, co_ratio1, co_ratio2, co_ratio3 = get_coincidence(array1, array2, array3)
    
    print "ave of the same columns: %d (%.3f, %.3f, %.3f)" % (co_num, co_ratio1, co_ratio2, co_ratio3)
    

# -----------------------------------------------------------
def clear_layer(net, layer, ABS_AVE_THRESHOLD):
    w = net.params[layer][0].data
    w.shape = [w.shape[0], -1]
    w_abs = np.abs(w)

    # clear columns
    index_of_pruned_column = np.where(np.average(w_abs, axis = 0) < ABS_AVE_THRESHOLD)[0] 
    w[:, index_of_pruned_column] = 0
    
    # clear rows
    clear_row = True
    if clear_row == True:
        index_of_pruned_row = np.where(np.average(w_abs, axis = 1) < ABS_AVE_THRESHOLD)[0] 
        w[:, index_of_pruned_row] = 0
    
    # save
    w.shape = net.params[layer][0].data.shape
    net.params[layer][0].data[:] = w[:]
    print "layer {} cleared, ABS_AVE_THRESHOLD = {}, clear_row = {}".format(layer, ABS_AVE_THRESHOLD, clear_row)
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

def arg_parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type = str, help = 'weights')
    parser.add_argument('--model', type = str, help = 'model', default = 'cifar10_full.prototxt')
    parser.add_argument('--PruneRate', type = float, help = 'PruneRate', default = 0)
    parser.add_argument('--layer', type = str, help = 'layer name', default = None)
    parser.add_argument('--ABS_AVE_THRESHOLD', type = float, help = 'ABS_AVE_THRESHOLD', default = 0)
    return parser.parse_args(args)
    
def main():
    args = arg_parse(sys.argv[1:])
    # prune net according to PruneRate
    if args.PruneRate > 0: 
        assert args.layer == None and args.ABS_AVE_THRESHOLD == 0
        prune_unimportant_columns(weights = args.weights, 
                                PruneRate = args.PruneRate,
                                    model = args.model)
    # prune net according to ABS_AVE_THRESHOLD
    elif args.ABS_AVE_THRESHOLD > 0 and args.layer == None:
        clear_minimals(weights = args.weights,
                         model = args.model,
             ABS_AVE_THRESHOLD = args.ABS_AVE_THRESHOLD)
    # check minimals of layer, according to ABS_AVE_THRESHOLD
    elif args.layer and args.ABS_AVE_THRESHOLD > 0:
        check_minimals(weights = args.weights, 
                         layer = args.layer, 
             ABS_AVE_THRESHOLD = args.ABS_AVE_THRESHOLD,
                         model = args.model)

 

if __name__ == "__main__":
    main()






