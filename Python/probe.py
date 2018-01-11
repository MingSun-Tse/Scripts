import sys
CAFFE = "/home/wanghuan/Caffe/Caffe_default/python"
sys.path.insert(0, CAFFE)
import caffe as c
import time
ctime = time.ctime().split(" ")[3]
import numpy as np


model = sys.argv[1]
weights = sys.argv[2]
net = c.Net(model, weights, c.TEST)


for layer, param in net.params.iteritems():
    # if layer != "conv2_1": continue
    # np.savetxt(layer + ".txt", param[0].data,  fmt = "%s", delimiter = "\n")
    print layer, np.abs(param[0].data).sum()
 

