from __future__ import print_function
import os
import sys
import shutil
import numpy as np
home = os.environ["HOME"]
caffe_root = home + "/Caffe/Caffe_default/python"
print (caffe_root)
sys.path.insert(0, caffe_root)

def my_move(file, dir):
    if type(file) == type(dir) == type("string"):
        if os.path.exists(file) and os.path.exists(dir):
            f = os.path.split(file)[-1]
            if os.path.exists(os.path.join(dir, f)):
                os.remove(os.path.join(dir, f))
            shutil.move(file, dir)

  
def get_free_gpu():
    os.system("nvidia-smi > smi_log.txt")
    lines = [l.strip() for l in open("smi_log.txt")]
    gpu_id = "0"
    min_mem_used_ratio = 1
    for i in range(len(lines)):
        if "MiB /" in lines[i]:
            mem_used  = float(lines[i].split("MiB")[0].split("|")[2].strip())
            mem_total = float(lines[i].split("/")[2].split("MiB")[0].strip())
            if mem_used / mem_total < min_mem_used_ratio:
                min_mem_used_ratio = mem_used * 1.0 / mem_total
                gpu_id = lines[i-1][4] ## assume less than 11 gpus, namely id = 0 - 9.
    os.remove("smi_log.txt")
    return gpu_id

    
def get_test_batch_size(model):
    assert os.path.isfile(model)
    cnt = 0
    for l in open(model):
        if "batch_size" in l:
            cnt += 1
        if cnt == 2:
            return int(l.split(":")[1].strip())
    print ("Error: train or test batch_size is not defined, please chech your net ptototxt.")
    return 0


def get_netproto(dir):
    assert dir.endswith("weights") or dir.endswith("weights/")
    if dir.endswith('/'):
        return dir[:-8] + "train_val.prototxt"
    else:
        return dir[:-7] + "train_val.prototxt"

def get_lr(acc_f, iter = None):
    lr = "NotFound"
    if iter:
        words = "Iteration %d, lr =" % iter
        for l in open(acc_f):
            if words in l:
                lr = l.split(words)[-1].strip()
    else:
        for l in open(acc_f):
            if "lr" in l:
                lr = l.split(" ")[-1].strip()
    return lr

    
def get_best_val(acc_file, criterion = "top5"):
    '''
        Choose the best validation result for next stage retraining. The val acc must be up then down, then choose the iter of peak acc.
        Rule of choice: If after 3k iters val_acc is still lower than the peak value, view this case as overfitting. Stop and prepare to decrease learning rate.
    '''
    acc = [] ## example: [[iter_1, acc1_1, acc5_1], [iter_2, acc1_2, acc5_2], ...]
    for l in open(acc_file):
        if len(l.split("  ")) != 3: continue ## skip non-acuracy lines
        acc.append([int(l.split("  ")[0].split("_")[2].split(".")[0]), float(l.split("  ")[1]), float(l.split("  ")[2])])
    
    acc = np.array(acc)
    dim = 1 if criterion == "top1" else 2
    max_acc = max(acc[:, dim])
    ix = np.where(acc[:, dim] == max_acc)[0]
    print ("the iter of max val acc is:", acc[ix][0][0])
    numk_iter_to_prob = 3
    if len(acc) >= ix + numk_iter_to_prob:
        print ("overfitting, stop at iter %s" % acc[ix][0][0])
        return int(acc[ix][0][0])
    return 0
       
       
def get_pid(f):
    lines = open(f).readlines()
    num_line = len(lines)
    for k in range(1, num_line + 1):
        pid = lines[-k].split(" ")[2]
        # check whether the pid I got is right
        if pid.isdigit():
            os.system("nvidia-smi | grep %s > pid.log" % pid)
            if open("pid.log").readline() != '':
                os.remove("pid.log")
                return pid
    print ("Error: cannot get pid")
    return
    

    
    