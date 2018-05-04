from __future__ import print_function
import os
import sys
import shutil
import numpy as np

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
            return int(l.split(":")[1].split("#")[0].strip())
    print ("Error: train or test batch_size is not defined, please chech your net ptototxt.")
    exit(1)

def change_test_batch_size(model, batch_size):
    assert os.path.isfile(model)
    output_file = model.replace(".prototxt", "_test-batch-size=" + str(batch_size) + ".prototxt")
    output = open(output_file, "w+")
    cnt = 0
    for line in open(model):
        newline = line
        if "batch_size" in line:
            cnt += 1
            if cnt == 2:
                newline = line.split("batch_size")[0] + "batch_size: " + str(batch_size) + "\n"
        output.write(newline)
    output.close()
    assert(cnt == 2)
    return output_file

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

def get_loss_acc(logFile):
    '''log example:
        I0502 16:36:55.485412  5246 caffe.cpp:325] loss = 1.52132 (* 1 = 1.52132 loss)
        I0502 16:36:55.485425  5246 caffe.cpp:325] top-1-accuracy = 0.6375
        I0502 16:36:55.485433  5246 caffe.cpp:325] top-5-accuracy = 0.8875
    '''
    lines = open(logFile).readlines()
    num_line = len(lines)
    loss, acc1, acc5 = -1, -1, -1
    for i in range(num_line, num_line - 10, -1): # In the last 10 lines, there should be loss, acc1 and acc5.
        line = lines[i].lower()
        if "loss = " in line and loss == -1:
            loss = float(line.split("loss = ")[1].split(" ")[0])
        if "acc" in line and " = " in line and acc1 == -1 and acc5 == -1:
            if "5" in line.split(" = ")[0].split("]"):
                acc5 = float(line.strip().split(" = ")[1])
            else:
                acc1 = float(line.strip().split(" = ")[1])
    return loss, acc1, acc5
    
    
    