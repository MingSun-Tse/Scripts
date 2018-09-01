#! /usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import sys
import numpy as np
import math
from optparse import OptionParser
from sklearn.decomposition import PCA
import shutil
import stat
import atexit
import time
try:
  import cPickle as pkl
except:
  import pickle as pkl

# Set Caffe
CAFFE_ROOT = os.path.abspath("./Caffe_APP")
sys.path.insert(0, os.path.join(CAFFE_ROOT, "python"))
import caffe

def prune_print(line):
  print("[app] %s  " % time.strftime("%Y/%m/%d-%H:%M") + str(line))

# Strip and overlook comments marked by '#'
def sharp_strip(line):
  return line.split("#")[0].strip()

def change_batch_size(model, out_model, which_batch, value):
  lines = open(model).readlines()
  if model == out_model:
    os.remove(model)
  out = open(out_model, "w+")
  for i in range(len(lines)):
    new_line = lines[i]
    if beginwith_str(sharp_strip(lines[i]), "batch_size"):
      # Get phase
      k = 1
      while not beginwith_str(sharp_strip(lines[i-k]), "phase"):
        k += 1
      phase = sharp_strip(lines[i-k]).split(":")[1].strip()
      if phase == which_batch:
        new_line = lines[i].split("batch_size")[0] + "batch_size: %s\n" % value
    out.write(new_line)
  out.close()
  
def add_prune_param(model, out_model):
  lines = open(model).readlines()
  out = open(out_model, 'w+')
  for i in range(len(lines)):
    new_line = lines[i]
    if "convolution_param" in lines[i]:
      blank = lines[i].split("convolution")[0]
      new_line = "".join([blank, "prune_param {\n",
                          blank, "  prune_ratio: 0\n",
                          blank, "  prune_ratio_step: 0\n",
                          blank, "}\n", 
                          lines[i]])
    out.write(new_line)
  out.close()

def set_prune_ratio(pr_file, model, out_model):
  # Find prune ratio from PCA
  pruned_ratio = {}
  for line in open(pr_file):
    if "pruning ratio of" in line:
      layer = line.split("of")[1].split(":")[0].strip()
      ratio = float(line.split(":")[1].split("(")[0].strip())
      assert(layer not in pruned_ratio.keys())
      pruned_ratio[layer] = ratio

  # Set prune_ratio in prototxt
  out_model = open(out_model, "w+")
  lines = [i for i in open(model)]
  for i in range(len(lines)):
    new_line = lines[i]
    if "prune_ratio:" in lines[i]:
      if for_acc:
        k = 1
        while "name" not in lines[i-k]: k += 1
        layer = lines[i-k].split('"')[1]
        prune_ratio = pruned_ratio[layer] if layer in pruned_ratio.keys() else 0
      else:
        prune_ratio = 0.95 # Set to a large value, which is usually not accessed
      new_line = lines[i].split("prune_ratio")[0] + "prune_ratio: %s\n" % prune_ratio # Set prune_ratio to a large ratio
    if "prune_ratio_step:" in lines[i]:
      k = 1
      while "name" not in lines[i-k]: k += 1
      layer = lines[i-k].split('"')[1]
      if layer in pruned_ratio.keys():
        new_line = lines[i].split("prune_ratio_step")[0] + "prune_ratio_step: " + str(pruned_ratio[layer]) + "\n"
    out_model.write(new_line)
  out_model.close()
  prune_print("Set prune ratio done")


def pca_analysis(w, keep_ratio):
  w = np.array(w)
  num_row, num_col = w.shape
  if num_row <= num_col:
    num_row, num_col = num_col, num_row # keep num_row is the number of example, must larger than number of attribute
    w = w.T
  n_components = int(np.ceil(keep_ratio * num_col))
  pca = PCA(n_components = n_components)

  # Substract mean
  mean = np.average(w, axis = 0)
  w = w - mean
  
  # PCA decomposition and weight reconstruction
  coff = pca.fit_transform(w)
  eigen = pca.components_
  w_ = coff.dot(eigen) # reconstructed weights
  
  # Get error ratio
  err = w - w_ # error vector
  err_ratio = 0
  for i in range(num_row):
    err_ratio += np.dot(err[i], err[i]) / np.dot(w[i], w[i])
  err_ratio /= num_row # average error ratio on all examples
  
  return err_ratio

def get_net_name(model):
  for line in open(model):
    line = sharp_strip(line)
    if beginwith_str(line, "name"):
      return line.split('"')[1]

def save_weights_to_npy(model, weights, weights_dir):
  GFLOPs = {}
  FLAG = caffe.TRAIN if "train" in model else caffe.TEST
  net = caffe.Net(model, weights, FLAG)
  for layer, param in net.params.iteritems():
    w = param[0].data
    if len(w.shape) != 4: continue # only save weights of conv layers
    w.shape = w.shape[0], -1
    np.save(os.path.join(weights_dir, "weights_%s.npy" % layer), w)
    GFLOPs[layer] = net.blobs[layer].shape[-2] * net.blobs[layer].shape[-1] * param[0].count # blob shape: [num, channel, height, width]
  with file(os.path.join(weights_dir, "GFLOPs.pkl"), 'w') as f:
    pkl.dump(GFLOPs, f)
  prune_print("Save weights and GFLOPs done")
  return GFLOPs

def get_layer_by_type(model, type):
  layers = []
  lines = [sharp_strip(i) for i in open(model)]
  for i in range(len(lines)):
    if beginwith_str(lines[i], "type") and lines[i].endswith('"' + type + '"'): # Note: ONLY support the new-version Caffe prototxt here.
      # Search layer name in neighbour lines
      upsearch_failed = False; downsearch_failed = False
      k = -1 # line offset
      while not beginwith_str(lines[i+k], "name"): # search upward
        k -= 1
        if beginwith_str(lines[i+k], "layer"):
          upsearch_failed = True
          k = 0
          break
      if upsearch_failed: # don't find layer name upward, so search downward
        k = 1
        while not beginwith_str(lines[i+k], "name"):
          k += 1
          if beginwith_str(lines[i+k], "layer"):
            downsearch_failed = True
            k = 0
            break
      if k == 0:
        prune_print("Wrong: the '%s' type layer in '%s' (Line %d) doesn't have layer name, please check." % (type, model, i))
        exit(1);
      else:
        layers.append(lines[i+k].split('"')[1])
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

def assign_prune_ratio(model, weights, speedup, exempted_layers = "", GFLOPs_weight = 0.8, keep_ratio_step = 0.1):
    assert(speedup > 0 and 0 <= GFLOPs_weight <= 1)
    exempted_layers = exempted_layers.split("/") if exempted_layers else []
    
    # Setup
    model = os.path.abspath(model)
    weights = os.path.abspath(weights)
    net_name = get_net_name(model)
    layers = get_layer_by_type(model, "Convolution")
    prune_print("net: " + net_name)
    prune_print("%s conv layers: %s" % (len(layers), layers))
    
    # Get GFLOPs
    weights_dir = os.path.join(os.path.split(model)[0], net_name)
    if not os.path.exists(weights_dir):
      os.makedirs(weights_dir)
    if os.path.exists(os.path.join(weights_dir, "GFLOPs.pkl")):
      with file(os.path.join(weights_dir, "GFLOPs.pkl"), 'r') as f:
        GFLOPs = pkl.load(f)
    else:
      GFLOPs = save_weights_to_npy(model, weights, weights_dir)
    prune_print("GFLOPs: %.4f MAC" % (np.sum(GFLOPs.values()) / 1e9))
    GFLOPs_ratio = {}
    for layer in layers:
      GFLOPs_ratio[layer] = GFLOPs[layer] * 1.0 / np.sum(GFLOPs.values())
    for layer in layers:
      if GFLOPs_ratio[layer] / np.average(GFLOPs_ratio.values()) <= 0.1 and GFLOPs_ratio[layer] <= 0.01: # automatically ignore those layers with little computation
        exempted_layers.append(layer)
    # prune_print("before twist, pr_based_on_GFLOPs:", GFLOPs_ratio)
    pr_based_on_GFLOPs = softmax(GFLOPs_ratio, exempted_layers)
    # prune_print("after  twist, pr_based_on_GFLOPs:", pr_based_on_GFLOPs)
      
    # PCA analysis
    prune_print("----------- PCA and GFLOPs analysis begin -----------")
    pr_based_on_PCA = {}
    if 0 <= GFLOPs_weight < 1:
      err_ratios = {}
      MIN, MAX, STEP = 0.1, 1, keep_ratio_step # the keep ratio range of PCA
      for layer in layers:
        if layer in exempted_layers:
          prune_print("%s exempted" % layer)
          continue
        err_ratios[layer] = []
        w = os.path.join(weights_dir, "weights_" + layer + ".npy")
        for keep_ratio in np.arange(MIN, MAX, STEP):
            err_ratios[layer].append(pca_analysis(np.load(w), keep_ratio))
        prune_print("%s PCA done" % layer)
      
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
      # prune_print("before twist, pr_based_on_PCA:", pr_based_on_PCA)
      pr_based_on_PCA = softmax(pr_based_on_PCA) # twist the ratios to make them more normal
      # prune_print("after  twist, pr_based_on_PCA:", pr_based_on_PCA)
    
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
    pr_file = os.path.join(weights_dir, "pruning_ratio_result_speedup_%s.txt" % speedup)
    with open(pr_file, 'w+') as f:
      for layer in layers:
        prune_ratio = 0 if layer in exempted_layers else multiplier * relative_prune_ratio[layer]
        line1 = "pruning ratio of %s: %.4f" % (layer, prune_ratio)
        line2 = " (GFLOPs ratio = %.3f)" % GFLOPs_ratio[layer]
        f.write(line1 + line2 + "\n")
        prune_print(line1 + line2)
        if prune_ratio >= 1:
          prune_print("BUG: pruning ratio >= 1")
    prune_print("----------- PCA and GFLOPs analysis end -----------")
    return pr_file

def beginwith_str(line, str):
  return str == line[:len(str)]
  
def set_up_solver(solver, out_solver):
  out = open(out_solver, "w+")
  cwd = os.getcwd(); os.chdir("../"); cwd_up = os.getcwd(); os.chdir(cwd)
  prune_wd = 0; wd = 0
  for line in open(solver):
    new_line = line
    line = sharp_strip(line)
    # Basic setting and safty check
    if beginwith_str(line, "max_iter"): new_line = 'max_iter: 100000000\n'
    if beginwith_str(line, "lr_policy"): new_line = 'lr_policy: "fixed"\n'
    if beginwith_str(line, "test_interval"): new_line = 'test_interval: 100000000\n'
    if beginwith_str(line, "snapshot") and (not beginwith_str(line, "snapshot_")): new_line = line if snapshot_when_prune else ''
    if beginwith_str(line, "snapshot_format"): new_line = '' # do not use HDF5 format, because the solverstate for pruning has not been implemented.
    if beginwith_str(line, "snapshot_prefix"): new_line = 'snapshot_prefix: "%s/weights/"\n' % cwd_up
    if beginwith_str(line, "net"): new_line = '            net: "%s/train_val.prototxt"\n' % cwd_up
    if beginwith_str(line, "solver_mode"): new_line = "solver_mode: GPU\n" # in case that user set this to CPU by mistake
    # Set base_lr
    if beginwith_str(line, "base_lr"):
      original_base_lr = float(line.split(":")[1].strip())
      if (original_base_lr >= 0.1 or original_base_lr <= 0.001):
        respond = raw_input("Provided 'base_lr' is uncommonly large/small (%s), are you sure it's right? (y/n) " % original_base_lr)
        if str.upper(respond) not in ["Y", "YES"]: exit(0)
      prune_base_lr = original_base_lr/20 # This setting is intuitive.
      new_line = "base_lr: %s\n" % prune_base_lr
    # Reset below
    if beginwith_str(line, "regularization_type"):
      new_line = ''
    # Reset below
    if beginwith_str(line, "weight_decay"):
      wd = float(line.split(":")[1].strip())
      new_line = ''
    # Reset below
    if beginwith_str(line, "iter_size"):
      new_line = ''
    out.write(new_line)
    
  # Check weight_decay
  new_wd = 0.5 * wd
  prune_wd = 0.5 * wd if wd else 1.0 / STD_PRUNE_TIME
  prune_wd = max(min(prune_wd, 5.0 / STD_PRUNE_TIME), 0.1 / STD_PRUNE_TIME) # constrain the std_target_reg in [0.1, 5.0]
  
  # Add new settings
  out.write('\n# Prune Setting -------------------------------------\n')
  out.write('prune_method: "Reg_Col"\n')
  out.write('regularization_type: "SelectiveReg"\n')
  out.write('weight_decay: %s\n\n' % new_wd)
  
  out.write('AA: %s\n' % prune_wd)
  out.write('target_reg: %s\n' % (prune_wd * STD_PRUNE_TIME))
  out.write('speedup: 10000\n')
  out.write('compRatio: 10000\n\n')
  
  out.write('iter_size: %s\n' % iter_size_in_solver_)
  out.write('iter_size_prune: 1\n')
  out.write('iter_size_losseval: %s\n' % iter_size_retrain_)
  out.write('iter_size_retrain: %s\n'  % iter_size_retrain_)
  out.write('iter_size_final_retrain: %s\n\n' % iter_size_final_retrain_)

  out.write('baseline_acc: %s\n' % baseline_acc_)
  out.write('acc_borderline: %s\n' % (baseline_acc_ - TOLERABLE_ACC_GAP))
  out.write('losseval_interval: %s\n' % LOSSEVAL_INTERVAL)
  out.write('retrain_test_interval: %s\n\n' % RETRAIN_TEST_INTERVAL)
  
  out.write('IF_eswpf: false\n')
  out.write('IF_speedup_count_fc: false\n')
  out.write('IF_update_row_col: false\n')
  out.write('IF_scheme1_when_Reg_rank: false\n')
  out.write('# ---------------------------------------------------')
  out.close()

def set_iter_size(solver, model):
  # Get iter from solver
  iter_size = 1 # 'iter_size' in the solver.prototxt
  for line in open(solver):
    line = sharp_strip(line)
    if beginwith_str(line, "iter_size"):
      iter_size = int(line.split(":")[1].strip())
      break
  
  # Get max_batch_size from model
  max_batch_size = 0 # the max batch_size a GPU card can support
  lines = open(model).readlines()
  for i in range(len(lines)):
    line = sharp_strip(lines[i])
    if beginwith_str(line, "phase") and line.endswith("TRAIN"):
      k = 1
      while not beginwith_str(sharp_strip(lines[i + k]), "batch_size"):
        k += 1
      max_batch_size = int(sharp_strip(lines[i + k]).split(":")[1].strip())
      break
  
  # Set iter_size
  global iter_size_retrain_, iter_size_final_retrain_, batch_size_in_model_, iter_size_in_solver_
  B = iter_size * max_batch_size # B: the real batch_size
  n = int(np.ceil(math.log(B, 2)))
  if n <= 5: # B <= 32
    b = 32
    iter_size_retrain_ = 1
    iter_size_final_retrain_ = 1
  elif 5 < n <= 8: # 32 < B <= 256
    b = 32
    iter_size_retrain_ = 2
    iter_size_final_retrain_ = 2**(n-5)
  else: # B > 256
    b = 2**(n - 3)
    iter_size_retrain_ = 2
    iter_size_final_retrain_ = 8
  
  # Handle b when a GPU cannot support b directly
  max_batch_size = int(2**np.floor(math.log(max_batch_size, 2))) # max batch_size of 2's
  batch_size_in_model_ = min(max_batch_size, b)
  iter_size_in_solver_ = max(b / max_batch_size, 1)

def set_up_train_sh(script, out_script, weights):
  out = open(out_script, "w+")
  cwd = os.getcwd(); os.chdir("../"); cwd_up = os.getcwd(); os.chdir(cwd)
  for line in open(script):
    new_line = line
    line = sharp_strip(line)
    if "CAFFE_ROOT" == line[:10]: new_line = 'CAFFE_ROOT="%s"\n' % CAFFE_ROOT
    if "PROJECT" == line[:7]: new_line = 'PROJECT="%s"\n' % cwd_up
    if "WEIGHTS" == line[:7]: new_line = 'WEIGHTS="%s"\n' % weights
    out.write(new_line)
  out.close()

def daemonize(script, original_dir):
  pid = os.fork()
  if pid: sys.exit(0)
  os.chdir('/')
  os.umask(0)
  os.setsid()
  _pid = os.fork()
  if _pid: sys.exit(0)
  
  # Flush I/O buffers
  sys.stdout.flush()
  sys.stderr.flush()
  
  os.chdir(original_dir)
  os.system(script)
  with open('/dev/null') as read_null, open('/dev/null', 'w') as write_null:
    os.dup2(read_null.fileno(), sys.stdin.fileno())
    os.dup2(write_null.fileno(), sys.stdout.fileno())
    os.dup2(write_null.fileno(), sys.stderr.fileno())

def test_once(model, weights, iters, gpu_id, return_acc1 = True):
  cwd = os.getcwd()
  os.chdir(CAFFE_ROOT)
  acc_log = ".%s_acc_log" % time.time()
  script = "build/tools/caffe test -model %s -weights %s -iterations %s -gpu %s 2>> %s" % (model, weights, iters, gpu_id, acc_log)
  os.system(script)
  
  # Get acc
  acc_layer = get_layer_by_type(model, "Accuracy")
  lines = open(acc_log).readlines()
  acc = []
  for i in range(len(lines)-1, -1, -1):
    output_name = lines[i].split("]")[1].split("=")[0].strip()
    if output_name in acc_layer:
      acc.append(float(lines[i].split("=")[1].strip()))
      if len(acc) == len(acc_layer):
        break
  assert(len(acc) == len(acc_layer))
  acc1, acc5 = min(acc), max(acc)
  assert(0 <= acc1 <= 1 and 0 <= acc5 <= 1)
  os.remove(acc_log)
  global baseline_acc_; baseline_acc_ = float(acc1) if return_acc1 else float(acc5)
  os.chdir(cwd)

def get_test_iter(solver):
  for line in open(solver):
    line = sharp_strip(line)
    if beginwith_str(line, "test_iter"):
      return int(line.split(":")[1].strip())

# ------------------------------------
# Global variables
iter_size_in_solver_ = 0
batch_size_in_model_ = 0
iter_size_retrain_ = 0
iter_size_final_retrain_ = 0
baseline_acc_ = 0

# Some settings
snapshot_when_prune = False
for_acc = False

# Constants (hyper-params)
TOLERABLE_ACC_GAP = 0.012
STD_PRUNE_TIME = 5000
LOSSEVAL_INTERVAL = 5000
RETRAIN_TEST_INTERVAL = 500
# ------------------------------------

if __name__ == "__main__":
  usage = \
  '''
  Example: python this_file -m user_train_val.prototxt -w user_pretrained.caffemodel -s user_solver.prototxt -g 0 -p user_project
  '''
  parser = OptionParser(usage = usage)
  parser.add_option("-s", "--solver", dest = "solver", type = "string", help = "the path of solver.prototxt")
  parser.add_option("-m", "--model", dest = "model", type = "string", help = "the path of train_val.prototxt")
  parser.add_option("-w", "--weights", dest = "weights", type = "string", help = "the path of caffemodel")
  parser.add_option("-p", "--project", dest = "project", type = "string", help = "the path of project directory")
  parser.add_option("-g", "--gpu", dest = "gpu", type = "string", help = "the GPU id", default = "0")
  parser.add_option("--for_acc", action = "store_true", dest="for_acc")
  parser.add_option("--for_pr", action = "store_false", dest="for_acc")
  parser.add_option("--speedup", dest = "speedup", type = "float", default = 2, help = "the target speedup when task at hand is 'given pr to get acc'")
  values, args = parser.parse_args(sys.argv)
  global for_acc;  for_acc = values.for_acc
  if not for_acc: assert(values.speedup == 2) # check

  # Set up project directory
  # use abspath to avoid not finding paths when changing directories later
  proj_path = os.path.abspath(values.project)
  solver_path = os.path.abspath(values.solver)
  model_path = os.path.abspath(values.model)
  weights_path = os.path.abspath(values.weights)

  if os.path.isdir(proj_path):
    respond = raw_input("The appointed project directory has existed. Do you want to overwrite it (all content inside will be removed)? (y/n) ")
    if str.upper(respond) in ["Y", "YES"]:
      shutil.rmtree(proj_path)
    else:
      exit(1)
  os.makedirs(proj_path)
  assert(os.path.isfile(solver_path) and os.path.isfile(model_path) and os.path.isfile(weights_path))
  prepare_dir = os.path.join(proj_path, ".prepare")
  os.makedirs(prepare_dir)
  os.makedirs(os.path.join(proj_path, "weights"))
  shutil.copyfile(solver_path, os.path.join(prepare_dir, "original_solver.prototxt"))
  shutil.copyfile(model_path, os.path.join(prepare_dir, "original_train_val.prototxt"))
  shutil.copyfile("train_template.sh", os.path.join(prepare_dir, "original_train.sh"))

  # Change directory
  # all the following changes will happen in the new directory
  os.chdir(prepare_dir)
  
  # Set iter_size (used in solver) and batch_size (used in model)
  set_iter_size(solver_path, model_path)
  
  # Set up train_val.prototxt
  change_batch_size("original_train_val.prototxt", "original_train_val_batch1.prototxt", "TRAIN", 1) # for use in assign_prune_ratio
  pr_file = assign_prune_ratio("original_train_val_batch1.prototxt", weights_path, values.speedup) # PCA and GFLOPs analysis
  add_prune_param("original_train_val.prototxt", "original_train_val_added_prune_param.prototxt")
  set_prune_ratio(pr_file, "original_train_val_added_prune_param.prototxt", "train_val.prototxt")
  change_batch_size("train_val.prototxt", "train_val.prototxt", "TRAIN", batch_size_in_model_)
  shutil.copyfile("train_val.prototxt", "../train_val.prototxt")
  prune_print("Set up 'train_val.prototxt' done")
  
  # Set up train.sh
  set_up_train_sh("original_train.sh", "train.sh", weights = weights_path)
  shutil.copyfile("train.sh", "../train.sh")
  os.chmod("../train.sh", 0o777) # empower the excution right
  prune_print("Set up 'train.sh' done")
  
  # Get baseline_acc_
  if for_acc:
    global baseline_acc_; baseline_acc_ = 0
  else:
    prune_print("Testing to get baseline accuracy")
    test_once(model_path, weights_path, get_test_iter(solver_path), values.gpu)
    prune_print("Testing done")
  # Set up solver.prototxt
  set_up_solver("original_solver.prototxt", "solver.prototxt")
  shutil.copyfile("solver.prototxt", "../solver.prototxt")
  prune_print("Set up 'solver.prototxt' done")
  
  # Run
  prune_print("Prepare done. Start pruning on GPU " + values.gpu)
  os.chdir("../")
  script = "./train.sh %s" % values.gpu
  daemonize(script, os.getcwd()) # run in daemon state
