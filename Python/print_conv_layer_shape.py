import sys
import os
pjoin = os.path.join

CAFFE_ROOT = "/home2/wanghuan/Caffe/Caffe_default"
sys.path.insert(0, pjoin(CAFFE_ROOT, "python"))
import caffe

model = sys.argv[1]
net = caffe.Net(model, caffe.TRAIN)

for layer, param in net.params.iteritems():
  if len(param[0].shape) != 4: continue
  w = param[0].data
  w.shape = w.shape[0], -1
  print('"%s": [%d, %d],' % (layer, w.shape[0], w.shape[1]))