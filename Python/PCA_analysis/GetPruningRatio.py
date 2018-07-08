#! /usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import sys
import numpy as np
import math
from optparse import OptionParser
try:
  import cPickle as pkl
except:
  import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set Caffe
HOME = os.environ["HOME"]
CAFFE_ROOT = os.path.join(HOME, "Caffe/Caffe_default")
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

def softmax(x, exempted_layers = []):
  values = []
  for k, v in x.items():
    if k in exempted_layers: continue
    values.append(pow(math.e, v))
  y = {}
  for k, v in x.items():
    if k in exempted_layers: continue
    y[k] = pow(math.e, v) / sum(values)
  return y

def main(model, weights, speedup, exempted_layers, GFLOPs_weight, keep_ratio_step):
    assert(speedup > 0 and 0 <= GFLOPs_weight <= 1)
    exempted_layers = exempted_layers.split("/") if exempted_layers else []
  
    # Setup
    model = os.path.abspath(model)
    weights = os.path.abspath(weights)
    net_name = get_net_name(model)
    layers = get_layers(model)
    print("net: " + net_name)
    print("%s conv layers:" % len(layers), layers)
    
    # Get GFLOPs
    weights_dir = os.path.join(os.path.split(model)[0], net_name)
    if not os.path.exists(weights_dir):
      os.mkdir(weights_dir)
    if os.path.exists(os.path.join(weights_dir, "GFLOPs.pkl")):
      with file(os.path.join(weights_dir, "GFLOPs.pkl"), 'r') as f:
        GFLOPs = pkl.load(f)
    else:
      GFLOPs = save_weights_to_npy(model, weights, weights_dir)
    print("GFLOPs: %.4f MAC" % (np.sum(GFLOPs.values()) / 1e9))
    GFLOPs_ratio = {}
    for layer in layers:
      GFLOPs_ratio[layer] = GFLOPs[layer] * 1.0 / np.sum(GFLOPs.values())
    for layer in layers:
      if GFLOPs_ratio[layer] / np.average(GFLOPs_ratio.values()) <= 0.1 and GFLOPs_ratio[layer] <= 0.01: # automatically ignore those layers with little computation
        exempted_layers.append(layer)
    # print("before twist, pr_based_on_GFLOPs:", GFLOPs_ratio)
    pr_based_on_GFLOPs = softmax(GFLOPs_ratio, exempted_layers)
    # print("after  twist, pr_based_on_GFLOPs:", pr_based_on_GFLOPs)
      
    # PCA analysis
    pr_based_on_PCA = {}
    if 0 <= GFLOPs_weight < 1:
      err_ratios = {}
      MIN, MAX, STEP = 0.1, 1, keep_ratio_step # the keep ratio range of PCA
      for layer in layers:
        if layer in exempted_layers:
          print("%s exempted" % layer)
          continue
        err_ratios[layer] = []
        w = os.path.join(weights_dir, "weights_" + layer + ".npy")
        for keep_ratio in np.arange(MIN, MAX, STEP):
            err_ratios[layer].append(pca_analysis(np.load(w), keep_ratio))
        print ("%s PCA done" % layer)
      
      for i in range(int((MAX - MIN) / STEP)):
        # relative prune_ratio is inversely proportional to err_ratio
        mul = 1.0
        for layer in layers:
          if layer in exempted_layers: continue
          mul *= err_ratios[layer][i]
        for layer in layers:
          if layer in exempted_layers: continue
          err_ratios[layer][i] = mul / err_ratios[layer][i] # now err_ratios is relative prune_ratio
        
        # normalize relative prune_ratio
        sum = 0.0
        for layer in layers:
          if layer in exempted_layers: continue
          sum += err_ratios[layer][i]
        for layer in layers:
          if layer in exempted_layers: continue
          err_ratios[layer][i] /= sum
          
      for layer in layers:
        if layer in exempted_layers: continue
        pr_based_on_PCA[layer] = np.average(err_ratios[layer])
      # print("before twist, pr_based_on_PCA:", pr_based_on_PCA)
      pr_based_on_PCA = softmax(pr_based_on_PCA) # twist the ratios to make them more normal
      # print("after  twist, pr_based_on_PCA:", pr_based_on_PCA)
    
    else:
      for layer in layers:
        pr_based_on_PCA[layer] = 0
   
    # combine PCA analysis and GFLOPs to determine the prune_ratio
    relative_prune_ratio = {}
    reduced_gflops = 0.0
    for layer in layers:
      if layer in exempted_layers: continue
      relative_prune_ratio[layer] = (1 - GFLOPs_weight) * pr_based_on_PCA[layer] + GFLOPs_weight * pr_based_on_GFLOPs[layer]
      reduced_gflops += relative_prune_ratio[layer] * GFLOPs[layer]
    multiplier = (np.sum(GFLOPs.values())  - np.sum(GFLOPs.values()) / speedup) / reduced_gflops
    
    # Check pruning ratio
    with open(os.path.join(weights_dir, "pruning_ratio_result_speedup_%s.txt" % speedup), 'w+') as f:
      for layer in layers:
        prune_ratio = 0 if layer in exempted_layers else multiplier * relative_prune_ratio[layer]
        line1 = "pruning ratio of %s: %.4f" % (layer, prune_ratio)
        line2 = " (GFLOPs ratio = %.3f)" % GFLOPs_ratio[layer]
        f.write(line1 + line2 + "\n")
        print(line1 + line2)
        if prune_ratio >= 1:
          print("bug: pruning ratio >= 1")

if __name__ == "__main__":
  usage = \
  '''
  usage example:
    python  this_file.py  -m deploy.prototxt  -w some_net.caffemodel
  for help:
    python  this_file.py  -h
  '''
  parser = OptionParser(usage = usage)
  parser.add_option("-m", "--model",   dest = "model",   type = "string", help = "the path of deploy.prototxt")
  parser.add_option("-w", "--weights", dest = "weights", type = "string", help = "the path of caffemodel")
  parser.add_option("-s", "--speedup", dest = "speedup", type = "float",  help = "speedup ratio you want (then '-s' > 1) or sparsity you want (then 0 < '-s' < 1)", default = 2.0)
  parser.add_option("-e", "--exempted_layers", dest = "exempted_layers", type = "string", help = "the layers not going to be pruned")
  parser.add_option("-g", "--GFLOPs_weight", dest = "GFLOPs_weight", type = "float", default = 0.8, help = "balance off the GFLOPs and PCA analysis when determining prune_ratio")
  parser.add_option("-k", "--keep_ratio_step", dest = "keep_ratio_step", type = "float", default = 0.1, help = "the keep_ratio_step in PCA analysis")
  values, args = parser.parse_args(sys.argv)
  if 0 < values.speedup <= 1:
    values.speedup = 1.0 / (1 - values.speedup) # in this way, '-s' means "--sparsity"
  main(values.model, values.weights, values.speedup, values.exempted_layers, values.GFLOPs_weight, values.keep_ratio_step)
