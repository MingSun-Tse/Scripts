#! /usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import sys
import numpy as np
from optparse import OptionParser
try:
  import cPickle as pkl
except:
  import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set Caffe
CAFFE_ROOT = "/home/wanghuan/Caffe/Caffe_default"
sys.path.insert(0, os.path.join(CAFFE_ROOT, "python"))
import caffe

def pca_analysis(w, keep_ratio):
  w = np.array(w)
  num_row, num_col = w.shape
  if num_row <= num_col:
    num_row, num_col = num_col, num_row
    w = w.T
  n_components = int(np.ceil(keep_ratio * num_col))
  pca = PCA(n_components = n_components)

  # substract mean
  mean = np.average(w, axis = 0)
  w = w - mean
  
  # PCA decomposition and weight reconstruction
  coff = pca.fit_transform(w)
  eigen = pca.components_
  w_ = coff.dot(eigen) # reconstructed weights
  
  err = w - w_
  err_ratio = 0
  for i in range(num_row):
    err_ratio += np.dot(err[i], err[i]) / np.dot(w[i], w[i])
  err_ratio /= num_row
  
  return err_ratio

def get_net_name(model):
  for line in open(model):
    if "name" in line:
      return line.split('"')[1]
      
def save_weights_to_npy(model, weights, weights_dir):
  GFLOPs = {}
  net = caffe.Net(model, weights, caffe.TEST)
  for layer, param in net.params.iteritems():
    w = param[0].data
    if len(w.shape) != 4: continue # only save weights of conv layers
    w.shape = w.shape[0], -1
    np.save(os.path.join(weights_dir, "weights_%s.npy" % layer), w)
    GFLOPs[layer] = net.blobs[layer].shape[-2] * net.blobs[layer].shape[-1] * param[0].count # blob shape: [num, channel, height, width]
  with file(os.path.join(weights_dir, "GFLOPs.pkl"), 'w') as f:
    pkl.dump(GFLOPs, f)
  print("save weights and GFLOPs done")
  return GFLOPs

def get_layers(model):
  layers = []
  lines = open(model).readlines()
  for i in range(len(lines)):
    if 'type' in lines[i] and '"Convolution"' in lines[i]: # only return conv layers
      k = 1
      while "name" not in lines[i - k]:
        k += 1
      layers.append(lines[i - k].split('"')[1])
  return layers

def main(model, weights, speedup, exempt_first_conv):
    assert(speedup > 0)
    # Setup
    model = os.path.abspath(model)
    weights = os.path.abspath(weights)
    net_name = get_net_name(model)
    print("net: " + net_name)
    weights_dir = os.path.join(os.path.split(model)[0], net_name)
    if not os.path.exists(weights_dir):
      os.mkdir(weights_dir)
    if os.path.exists(os.path.join(weights_dir, "GFLOPs.pkl")):
      with file(os.path.join(weights_dir, "GFLOPs.pkl"), 'r') as f:
        GFLOPs = pkl.load(f)
    else:
      GFLOPs = save_weights_to_npy(model, weights, weights_dir)
    print("GFLOPs: %.4f MAC" % (np.sum(GFLOPs.values()) / 1e9))
    
    # PCA analysis
    err_ratios = {}
    MIN, MAX, STEP = 0.1, 1, 0.05 # the keep ratio range of PCA
    layers = get_layers(model)
    print("%s conv layers:" % len(layers), layers)
    GFLOPs_exempted = 0 # the GFLOPs sum of exempted layers
    for layer in layers:
      if layer == layers[0] and exempt_first_conv:
        print("%s exempted" % layer)
        err_ratios[layer] = [0] * int((MAX - MIN) / STEP)
        GFLOPs_exempted += GFLOPs[layer]
        continue
      err_ratios[layer] = []
      w = os.path.join(weights_dir, "weights_" + layer + ".npy")
      for keep_ratio in np.arange(MIN, MAX, STEP):
          err_ratios[layer].append(pca_analysis(np.load(w), keep_ratio))
      print ("%s PCA done" % layer)
    
    # Calculate pruning ratio
    for i in range(int((MAX - MIN) / STEP)):
      sum = 0
      for layer in layers:
        sum += err_ratios[layer][i]
      for layer in layers:
        err_ratios[layer][i] /= sum
    remaining_ratio = {}
    sum_gflops = 0
    for layer in layers:
      remaining_ratio[layer] = np.average(err_ratios[layer])
      sum_gflops += remaining_ratio[layer] * GFLOPs[layer]
    multiplier = (np.sum(GFLOPs.values()) / speedup - GFLOPs_exempted) / sum_gflops
    
    # Check pruning ratio
    with open(os.path.join(weights_dir, "pruning_ratio_result_speedup=%s.txt" % speedup), 'w+') as f:
      for layer in layers:
        prune_ratio = 0 if layer == layers[0] and exempt_first_conv else 1 - multiplier * remaining_ratio[layer]
        line1 = "pruning ratio of %s: %.2f" % (layer, prune_ratio)
        line2 = " (GFLOPs ratio = %.3f)" % (GFLOPs[layer] * 1.0 / np.sum(GFLOPs.values()))
        f.write(line1 + line2 + "\n")
        print(line1 + line2)
        if multiplier * remaining_ratio[layer] > 1:
          print("bug: pruning ratio < 0")

if __name__ == "__main__":
  usage = \
  '''
  usage example:
    python  this_file.py  -m deploy.prototxt  -w some_net.caffemodel  -s 4
  for help:
    python  this_file.py  -h
  '''
  parser = OptionParser(usage = usage)
  parser.add_option("-m", "--model",   dest = "model",   type = "string", help = "the path of deploy.prototxt")
  parser.add_option("-w", "--weights", dest = "weights", type = "string", help = "the path of caffemodel")
  parser.add_option("-s", "--speedup", dest = "speedup", type = "float",  help = "speedup ratio you want")
  parser.add_option("-e", "--exempt_first_conv", dest = "exempt_first_conv", type = "int", default = True, help = "whether not to prune the first conv layer, default true")
  values, args = parser.parse_args(sys.argv)
  main(values.model, values.weights, values.speedup, values.exempt_first_conv)