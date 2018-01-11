#! /usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import print_function
import numpy as np
import os
import stat
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import shutil
import time

caffe_root = "/home/wanghuan/Caffe/Caffe_APP/"
sys.path.insert(0, caffe_root + 'python')
import caffe as c

from util import get_pid, get_best_val, get_test_batch_size
from test_acc import Tester

class ExpDeployer():
    def __init__(self, dir, lr, num, gpu_train, gpu_test):
        self.project_dir      = dir
        self.lr               = lr
        self.num_test_example = num
        self.gpu_train        = gpu_train
        self.gpu_test         = gpu_test
        self.model            = dir + "/train_val.prototxt"
        self.test_batch_size  = get_test_batch_size(self.model)
        self.timeID           = None
        self.caffe_root       = dir.split("MyProject")[0]
        self.work_dir         = dir.split("MyProject")[1]
        self.stage            = 0
        self.turning_iter     = []
    
    def get_iter_prune_finish(self):
        prune_log = "weights/log_%s_prune.txt" % self.timeID
        for l in open(prune_log):
            if "all layer prune finish" in l: ## example: `all layer prune finish: 34566`
                os.system("cat %s | grep 'prune finish' | awk '{print $1, $5}' > prune.log" % prune_log)
                return int(l.split(" ")[-1])
        return 0

    def get_pretrained_model(self):
        w = [i for i in os.listdir(self.project_dir + "/weights") if ".caffemodel" in i ]
        if w:
            return w[0]
        print ("Error: cannot find pretrained model")
        return w[0]
        
    def make_script(self, stage):
        if not self.timeID:
            self.timeID = time.strftime("%Y%m%d-%H%M")
        
        if stage == 0:
            file_in  = "template/train.sh"
            file_out = "train.sh"
        else:
            file_in  = "template/retrain.sh"
            file_out = "retrain_%d.sh" % stage
        fi = open(file_in, 'r')
        fo = open(file_out, "w+")
        
        for l in fi:
            new_l = l
            if "CAFFE_ROOT=" in l:
                new_l = "CAFFE_ROOT=%s\n" % self.caffe_root
            if "WORK_DIR=" in l:
                new_l = "WORK_DIR=%s\n" % self.work_dir
            if "TIME=" in l:
                new_l = "TIME=%s\n" % self.timeID
            if "-gpu" in l:
                new_l = "-gpu %s\n" % self.gpu_train
            if "-solver" in l:
                if "ToBeReplaced" in l:
                    new_l = l.replace("ToBeReplaced", "retrain_solver_%d.prototxt" % stage)
            if "-weights" in l:
                new_l = l.replace("ToBeReplaced", self.get_pretrained_model())
            if "-snapshot" in l:
                prefix = "" if stage == 1 else "retrain"
                so = "%s_iter_%d.solverstate" % (prefix, self.turning_iter[stage - 1])
                ca = "%s_iter_%d.caffemodel"  % (prefix, self.turning_iter[stage - 1]) 
                new_l = l.replace("ToBeReplaced", so)
                [os.system("mv  weights/tested_weights/%s  weights" % i) for i in [so, ca]] ## TO improve
            fo.write(new_l)
        fi.close()
        fo.close()
        os.chmod(file_out, stat.S_IRWXU) ## add excute for USER
    
    def make_solver(self, stage):
        if stage == 0:
            fi = open("template/train_solver.prototxt", 'r')
            fo = open("train_solver.prototxt", "w+")
        else:
            fi = open("template/retrain_solver.prototxt", 'r')
            fo = open("retrain_solver_%d.prototxt" % stage, "w+")
        for l in fi:
            new_l = l
            if "base_lr" in l:
                new_l = "base_lr: %s\n" % self.lr[stage]
            if "snapshot_prefix" in l:
                suffix = "" if stage == 0 else "retrain"
                new_l = l.replace("ToBeReplaced", self.project_dir + "/weights/" + suffix)
            if "net:" in l:
                new_l = l.replace("ToBeReplaced", self.model)
            fo.write(new_l)
        fi.close()
        fo.close()
        
    def deploy(self):
        os.chdir(self.project_dir)
        self.make_script(0)
        self.make_solver(0)
        os.system("nohup ./train.sh > /dev/null &")
        time.sleep(5)
        print ("%s - train/prune started" %  time.strftime("%Y%m%d-%H:%M:%S"))
        
        # Check if prune finished
        while 1:
            iter = self.get_iter_prune_finish()
            if iter:
                break
            print ("%s - prune not finished, I will check it again 1 minutes later..." % time.strftime("%Y%m%d-%H:%M:%S"))
            time.sleep(10)
        self.turning_iter.append(iter)
        print ("%s - prune finished, iter = %d" % (time.strftime("%Y%m%d-%H:%M:%S"), iter))
        [os.system("mv  weights/%s  weights/tested_weights" % i) for i in os.listdir("weights") if "caffemodel" in i or "solverstate" in i]
        
        # Retrain
        for stg in range(1, len(self.lr)):
            self.make_script(stg)
            self.make_solver(stg)
            os.system("nohup ./retrain_%d.sh > /dev/null &" % stg)
            time.sleep(5)
            print ("%s - the %dth retrain begins" % (time.strftime("%Y%m%d-%H:%M:%S"), stg))
            while (1):
                # if new caffemodel comes out, update val_accuracy
                new_caffemodel = [i for i in os.listdir("weights") if "caffemodel" in i]
                if new_caffemodel:
                    tester = Tester(self.model, self.project_dir + "/weights", self.num_test_example, self.gpu_test)
                    tester.test()
                    
                    # after test, check val_accuracy to see whether to stop and decrease learning rate
                    best_val_iter = get_best_val("weights/val_accuracy.txt")
                    if best_val_iter:
                        pid = get_pid("weights/log_%s_acc.txt" % self.timeID)
                        os.system("sudo kill -9 %s" % pid)
                        self.turning_iter.append(best_val_iter)
                        print ("%s - the %dth retrain comes to end, with the best_val_iter = %s, " % (time.strftime("%Y%m%d-%H:%M:%S"), stg, best_val_iter))
                        break
                print ("%s - retrain not finished, I will check it again 1 minitue later..." % time.strftime("%Y%m%d-%H:%M:%S"))
                time.sleep(10)
        print ("%s - pruning and retrain done!" % time.strftime("%Y%m%d-%H:%M:%S"))


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-project', '-p', type = str, help = "the project_dir")
    parser.add_argument('-lr', type = str, help = "learning rate for train and retrain")
    parser.add_argument('-num_test_example', '-n', type = int, default = 50000, help = "the number of examples to test")
    parser.add_argument('-gpu_train', '-g1', type = str, help = "GPU id for train")
    parser.add_argument('-gpu_test', '-g2', type = str, help = "GPU id for test")
    return parser.parse_args(args)

if __name__ == "__main__":
    '''
        python  deploy_exp.py \
            -p /home/wanghuan/Caffe/Caffe_APP_2/Caffe_APP/MyProject/NewExpsOnCifar10/13_PP_convnet_cifar10/4x_test_deploy \
            -lr 0.001-0.0001-0.00001 \
            -n 10000 \
            -g1 4 \
            -g2 1
    '''
    args = parse_args(sys.argv[1:])
    lr = [float(i) for i in args.lr.split('-')]
    deployer = ExpDeployer(args.project,
                           lr,
                           args.num_test_example,
                           args.gpu_train,
                           args.gpu_test)
    deployer.deploy()

    
