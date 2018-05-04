#! /usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import print_function
import numpy as np
import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import shutil
import time

# set caffe
HOME = os.environ["HOME"]
caffe_root = os.path.join(HOME, "Caffe/Caffe_default")
# if not os.path.isdir(caffe_root):
    # caffe_root = os.path.join(HOME, "Caffe/Caffe_Compression")
    # if not os.path.isdir(caffe_root):
        # print ("caffe_root doesn't exist, please check.")
        # exit(1)
print ("using caffe @ '%s'" % caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe as c
from util import get_free_gpu, get_test_batch_size, change_test_batch_size, get_netproto, get_lr, my_move, get_loss_acc

class Tester():
    def __init__(self, model, dir, num, gpu, batch_size):
        self.model            = model if model else get_netproto(dir)
        self.weight_dir       = dir
        self.num_test_example = num
        self.gpu_id           = gpu if gpu else get_free_gpu()
        self.test_batch       = batch_size if batch_size != 0 else get_test_batch_size(self.model)
        self.model            = change_test_batch_size(self.model, batch_size) if batch_size != 0 else self.model
        tmp                   = [i for i in os.listdir(dir) if "retrain_acc.txt" in i]
        self.acc_log          = os.path.join(dir, tmp[0]) if len(tmp) else "does_not_exist_hhh"
    
    
    def test_once(self, weights):
        test_iter = int(np.ceil(float(self.num_test_example) / self.test_batch))
        print ("test_iter is:", test_iter)
        tt = int(time.time())
        out_log = "acc_log_%s_%s.txt" % (tt, weights.replace(os.sep, "_"))
        if out_log in os.listdir("."):
            print("Wrong: naming conflict, there is already an accuracy log file named `%s` in the current directory." % out_log)
            exit(1)
        script = "".join([caffe_root, "/build/tools/caffe test",
                                      " -model ", self.model,
                                      " -weights ", weights, 
                                      " -gpu ", self.gpu_id,
                                      " -iterations ", str(test_iter), 
                                      " 2> ", out_log])
        os.system(script)
        
        lines = open(out_log).readlines()
        acc1 = lines[-2].strip().split(" ")[-1]
        acc5 = lines[-1].strip().split(" ")[-1] # TODO(Ming): fix this for what has no acc5
        
        loss, acc1, acc5 = get_loss_acc(out_log)
        os.remove(out_log) 
        print ("loss and acurracy: " + str(loss) + "  " + str(acc1) + "  " + str(acc5))
        return loss, acc1, acc5
        

    def test(self):
        IF_has_retrain = sum(["retrain" in i for i in os.listdir(self.weight_dir) if ".caffemodel" in  i]) > 0 # if has "retrain", only eval "retrain" cafffemodel's acc
        name_mark = "retrain" if IF_has_retrain else "e" # use 'e' because 'caffemodel' and 'solverstate' both have 'e'
        iters = [int(i.split("_")[-1].split(".")[0]) for i in os.listdir(self.weight_dir) if ".caffemodel" in i and name_mark in i]
        iters.sort()
        print("all the caffemodels:\n", iters)
        
        # For tested caffemodels, move them to `tested_weights`
        tested_weight_dir = os.path.join(self.weight_dir, "tested_weights")
        if not os.path.isdir(tested_weight_dir):
            os.makedirs(tested_weight_dir)
      
        while len(iters):
            iter = iters.pop(0)
            print ("dealing with iter %s's caffemodel:" % iter, sep=" ")
            acc_file = os.path.join(self.weight_dir, "val_accuracy.txt")
            fp = open(acc_file, "a+")
            
            weights      = [os.path.join(self.weight_dir, i) for i in os.listdir(self.weight_dir) if "_iter_"+str(iter)+".caffemodel"  in i and name_mark in i][0]
            solverstates = [os.path.join(self.weight_dir, i) for i in os.listdir(self.weight_dir) if "_iter_"+str(iter)+".solverstate" in i and name_mark in i]
            solverstate  = solverstates[0] if len(solverstates) else None
            print (weights.split(os.sep)[-1])
            loss, acc1, acc5 = self.test_once(weights)
           
            if os.path.exists(self.acc_log):
                lr = get_lr(self.acc_log, iter)
                if lr != get_lr(acc_file):
                    fp.write("\nlr = " + lr + "\n")
            
            line = "".join("%-30s" % weights.split(os.sep)[-1],
                            "  %7.5f" % loss,
                            "  %7.5f" % acc1,
                            "  %7.5f\n" % acc5)
            fp.write(line)
            fp.close()
            my_move(weights, tested_weight_dir)
            my_move(solverstate, tested_weight_dir)
            
            # update iters pool
            iters = [int(i.split("_")[-1].split(".")[0]) for i in os.listdir(self.weight_dir) if ".caffemodel" in i and name_mark in i]
            iters.sort()
        
        if "_test-batch-size=" in self.model:
            os.remove(self.model)

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',            '-m', type = str, default = None,  help = "The prototxt of network definition")
    parser.add_argument('--weight_dir',       '-w', type = str,                  help = "The folder containing caffemodels of weights")
    parser.add_argument('--num_test_example', '-n', type = int, default = 50000, help = "the number of examples to test")
    parser.add_argument('--gpu',              '-g', type = str, default = None,  help = "GPU id")
    parser.add_argument('--batch_size',       '-b', type = int, default = 0,     help = "test batch size") # if batch_size not supplied, use the one in prototxt
    return parser.parse_args(args)

if __name__ == "__main__":
    '''Usage:
        (1) python  test_acc.py  -m train_val.prototxt  -w weight_dir -g 2 -n 10000  # user supplies `train_val.prototxt`
        (2) python  test_acc.py  -w weight_dir                                       # automatically find `train_val.prototxt`
    '''
    args = parse_args(sys.argv[1:])
    tester = Tester(args.model, args.weight_dir, args.num_test_example, args.gpu, args.batch_size)
    tester.test()
    
