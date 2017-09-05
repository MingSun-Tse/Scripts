import sys
import numpy as np
caffe_root = "/home/wanghuan/Caffe/Caffe_default/"
sys.path.insert(0, caffe_root + 'python')
import caffe as c

def compute_pruned_row_of_alexnet(args):
    cols = []
    cols.append([ 96, 121, np.load(args[0])])
    cols.append([256,  25, np.load(args[1])])
    cols.append([384,   9, np.load(args[2])])
    cols.append([384,   9, np.load(args[3])])
    cols.append([256,   9, np.load(args[4])])

    for k in range(len(args) - 1):
        num_row = cols[k][0]
        filter_area = cols[k+1][1]
        cnt = 0
        pruned_rows = []
        for i in range(num_row):
            flag = 1
            for j in range(i * filter_area, (i+1) * filter_area):
                if not j in cols[k+1][2]:
                    flag = 0
                    break
            if flag: 
                pruned_rows.append(i)
                cnt += 1
        print "conv" + str(k+1) + " Total pruned rows:", cnt
        print pruned_rows    

        
group_ = {"0conv1":1, "1conv2":2, "2conv3":1, "3conv4":2, "4conv5":2}
def find_row_2(model, weights):
    net1 = c.Net(model, weights, c.TEST) # original net
    net2 = c.Net(model, weights, c.TEST) # will be row pruned
    layers = []
    conv_ws = []
    pruned_rows = {}
    
    for layer, param in net2.params.iteritems():
        if len(param[0].data.shape) == 4:
            layers.append(layer)
            conv_ws.append(param[0].data[:])
            pruned_rows[layer] = []
        
    for l in [2]: #range(len(layers) - 1) :
        num_row = conv_ws[l].shape[0]
        filter_area_next_layer = conv_ws[l+1].shape[2] * conv_ws[l+1].shape[3]
        w_next_layer = conv_ws[l+1][:]
        w_next_layer.shape = [w_next_layer.shape[0], -1]
        
        for i in range(num_row):
            i_ = i % (num_row / group_[layers[l+1]])
            flag = 1
            for j in range(i_ * filter_area_next_layer, (i_+1) * filter_area_next_layer):
                if sum(np.abs(w_next_layer[:, j])) != 0:
                    flag = 0
                    break
            if flag:
                conv_ws[l][i] = np.zeros(conv_ws[l].shape[1:])
                pruned_rows[layers[l]].append(i)
        net2.params[layers[l]][0].data[:] = conv_ws[l][:]
    print pruned_rows, len(pruned_rows["2conv3"])
    net2.save(weights.split(".caffemodel")[0] + "_rowpruned.caffemodel")
    if compare_net(net1, net2, "conv4") == 0:
        print "same"
    else:
        print "not same"
    return pruned_rows

def prune_row(model, weights, layer, rows):
    net1 = c.Net(model, weights, c.TEST)
    net2 = c.Net(model, weights, c.TEST)
    w_shape = net1.params[layer][0].data.shape
    num_row = w_shape[0]
    diff = []
    for r in rows:
        net2.params[layer][0].data[r] = np.zeros(w_shape[1:])
    net2.save(weights.split(".caffemodel")[0] + "_rowpruned.caffemodel")
    diff = compare_net(net1, net2, layer)
    if diff == 0:
        print "same"
    else:
        print "not same"
        
def compare_net(net1, net2, output_layer):
    image = "Adam2.jpg"
    imdata = c.io.load_image(image)
    result = []
    transformer = c.io.Transformer({'data': net1.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformed_image = transformer.preprocess('data', imdata)
    for net in (net1, net2):
        net.blobs["data"].data[...] = transformed_image
        net.forward()
        result.append(net.blobs[output_layer].data[0])
    difference = (np.abs((result[0] - result[1]))).sum()
    print difference
    return difference

    
def find_row(model, weights, layer):
    net1 = c.Net(model, weights, c.TEST) # original model
    net2 = c.Net(model, weights, c.TEST) # row-pruned model
    num_row = net1.params[layer][0].data.shape[0]
    rows = []
    for r in range(num_row):
        print "processing row %d..." % r
        net2.params[layer][0].data[:] = net1.params[layer][0].data[:]
        net2.params[layer][0].data[r] = np.zeros(net2.params[layer][0].data[r].shape)
        diff = compare_net(net1, net2, "conv5")
        if diff == 0:
            rows.append(r)
            print "row %d can be pruned" % r
    print "%d rows of %s can be pruned without influence:" % (len(rows), layer), rows
    return rows

def tmp(model, weights):
    image = "fish.png"
    imdata = c.io.load_image(image)
    net1 = c.Net(model, weights, c.TEST)
    net2 = c.Net(model, weights, c.TEST)
    transformer = c.io.Transformer({'data': net1.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformed_image = transformer.preprocess('data', imdata)

    net1.blobs["data"].data[...] = transformed_image
    net2.blobs["data"].data[...] = transformed_image
    
    layer = "0conv1"
    next_layer = str(int(layer[0])+1) + "conv" + str(int(layer[-1])+1)
    net1.params[layer][0].data[:] = np.ones(net1.params[layer][0].data.shape) * 0.0045
    net2.params[layer][0].data[:] = np.ones(net2.params[layer][0].data.shape) * 0.0045
    
    net1.params[next_layer][0].data[:]    = np.ones(net1.params[next_layer][0].data.shape) * 0.005
    net2.params[next_layer][0].data[:]    = np.ones(net2.params[next_layer][0].data.shape) * 0.005
    net1.params[next_layer][0].data[:,15:17,:,:] = np.zeros(net1.params[next_layer][0].data[:,15:17,:,:].shape)
    net2.params[next_layer][0].data[:,15:17,:,:] = np.zeros(net2.params[next_layer][0].data[:,15:17,:,:].shape) # set only channel pr to zeros

    
    for r in range(net1.params[layer][0].data.shape[0]):
        print "processing row %d..." % r
        net2.params[layer][0].data[:] = net1.params[layer][0].data[:]
        net2.params[layer][1].data[:] = net1.params[layer][1].data[:]
        
        net2.params[layer][0].data[r] = np.zeros(net2.params[layer][0].data[r].shape) # pruned row 1 - weights
        net2.params[layer][1].data[r] = np.zeros(net2.params[layer][1].data[r].shape) # pruned row 1 - biases
       
        net1.forward()
        net2.forward()    
        if np.abs(net1.blobs["conv5"].data[0] - net2.blobs["conv5"].data[0]).sum() == 0:
            print "   %d can be pruned" % r
    
   

 
    
    
if __name__ == "__main__":
    if len(sys.argv) == 6:
        compute_pruned_row_of_alexnet(sys.argv[1:])
    elif len(sys.argv) == 3:
        tmp(sys.argv[1], sys.argv[2])
        # layer = "0conv1"
        # rows = find_row(sys.argv[1], sys.argv[2], layer)
        # prune_row(sys.argv[1], sys.argv[2], layer, rows)