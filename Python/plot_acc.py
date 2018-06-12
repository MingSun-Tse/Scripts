from __future__ import print_function
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ColorSet = ("r", "b", "g", "k", "c", "y")
CntPlot  = 0
Nets = ("caffeNet", "cifar10_full")
Net  = "None"
ColNum = {"cifar10_full":(75, 800, 800), 
           "caffeNet":(363, 1200, 2304, 1728, 1728)} ## layer 2,4,5, group = 2

def plot_acc(log_file, ID):
    val_acc             = []
    val_acc5            = []
    val_loss            = []
    train_loss          = []
    train_loss_smoothed = []
    acc5_of_diff_lr_stage = {} # like the accuracies when lr = 0.005
    global CntPlot

    
    ## Parsing
    lines = [l.strip() for l in open(log_file)]
    IF_find_Net = False
    for i in range(len(lines)):
        ## get net name
        if (not IF_find_Net) and '"' in lines[i] and lines[i].lower().split('"')[1] in Nets: 
            global Net
            Net = lines[i].lower().split('"')[1]
            IF_find_Net = True
        
        
        ## val acc
        if "Test net output" in lines[i] and "accuracy = " in lines[i]:
            acc_ =  float(lines[i].split("= ")[-1])
            j = 1
            while(not lines[i-j].split("Iteration ")[-1].split(",")[0].isdigit()):
                j = j + 1
            iter_ = int(lines[i-j].split("Iteration ")[-1].split(",")[0])
            val_acc.append([iter_, acc_])
            
        
        ## val acc top5 
        if "Test net output" in lines[i] and "accuracy_top_5 = " in lines[i]:
            acc_ =  float(lines[i].split("= ")[-1])
            j = 1
            while(not lines[i-j].split("Iteration ")[-1].split(",")[0].isdigit()):
                j = j + 1
            iter_ = int(lines[i-j].split("Iteration ")[-1].split(",")[0])
            val_acc5.append([iter_, acc_])
            
            # get acc of each learning rate
            j = 1
            while(not "lr = " in lines[i+j]):
                j += 1
            lr_ = str(float(lines[i+j].split("lr = ")[-1])) # for '1e-4' -> '0.0001'
            if lr_ in acc5_of_diff_lr_stage.keys():
                acc5_of_diff_lr_stage[lr_].append([iter_, acc_])
            else:
                acc5_of_diff_lr_stage[lr_] = [[iter_, acc_]]
        
        ## val loss
        if "Test net output" in lines[i] and "loss" in lines[i]:
            k = 2
            while(not lines[i-k].split("Iteration ")[-1].split(",")[0].isdigit()):
                k = k + 1
            val_loss.append([int(lines[i-k].split("Iteration ")[-1].split(",")[0]),  float(lines[i].split("loss = ")[-1].split(" ")[0])])
        
        ## train loss
        if "Train net output" in lines[i] and "loss" in lines[i]: 
            t = 1
            while(not lines[i-t].split("Iteration ")[-1].split(",")[0].isdigit()):
                t = t + 1
            train_loss.append([int(lines[i-t].split("Iteration ")[-1].split(",")[0]),  float(lines[i].split("loss = ")[-1].split(" ")[0])]) 
        
        ## smoothed train loss
        if "Iteration" in lines[i] and "loss = " in lines[i]: 
            train_loss_smoothed.append([int(lines[i].split("Iteration ")[-1].split(",")[0]),  float(lines[i].split("loss = ")[-1].split(",")[0])])

            
    ## Converted to np array for convenience
    assert(len(val_acc) == len(val_loss))
    train_loss          = np.array(train_loss)
    train_loss_smoothed = np.array(train_loss_smoothed)
    val_loss            = np.array(val_loss)
    val_acc             = np.array(val_acc)
    val_acc5            = np.array(val_acc5)
    
    ## Print statistics
    print ("\ntimeID: %s" % ID)
    lrs = sorted([float(i) for i in acc5_of_diff_lr_stage.keys()], reverse=True)
    for lr in lrs:
        acc5 = np.array(acc5_of_diff_lr_stage[str(lr)])
        iter_begin = int(acc5[0, 0]) # the iter to begin this lr
        max_acc = max(acc5[:, 1]) # the max acc during this lr
        min_acc = min(acc5[:, 1]) # the min acc during this lr
        ave_acc = np.average(acc5[:, 1]) # the average acc during this lr
        print ("lr = {:7.5f}({}):  {:7.5f} - {:7.5f} - {:7.5f}".format(lr, iter_begin, min_acc, max_acc, ave_acc))
    
    if (len(val_acc5)):
        max_acc5 = float(("%4.10f" % max(val_acc5[:,1]))[:4]) # use `.10f`, in order not to round
        min_acc5 = float(("%4.10f" % min(val_acc5[:,1]))[:4])
        step = 0.01
        sep = "   "
        print ("acc_min", end=sep)
        for acc5 in np.arange(min_acc5, max_acc5, step):
            print ("%4.2f+" % acc5, end=sep)
        print ("acc_max")
        print ("%7.5f" % min(val_acc5[:,1]), end=sep)
        for acc5 in np.arange(min_acc5, max_acc5, step):
            num = len(np.where(np.logical_and(val_acc5[:,1]>=acc5,  val_acc5[:,1]<acc5+step))[0])
            print ("%5d" % num, end=sep)
        print ("%7.5f" % max(val_acc5[:,1]))

    
    ## Plot loss
    # plt.plot(train_loss[:,0], train_loss[:,1], "-k", label = "train loss")
    # train_loss_smoothed = smooth(train_loss_smoothed, 1000)
    # plt.plot(train_loss_smoothed[:,0], train_loss_smoothed[:,1], "-b", label = "smoothed train loss")
    
    # if len(val_loss):
        # plt.plot(val_loss[:,0], val_loss[:,1], "-", color = ColorSet[CntPlot%len(ColorSet)], label = "val_%s_loss"%ID); CntPlot += 1
    
    ## Plot val acc
    if len(val_acc):
	smooth_step = 1
        smoothed_val_acc = smooth(val_acc[:,1], smooth_step)
        myprint (diff(smoothed_val_acc * 100))
	label = "{}_val_acc, smooth_step={}".format(ID, smooth_step)
        plt.plot(val_acc[:,0], smoothed_val_acc[:], "-", color=ColorSet[CntPlot%len(ColorSet)], label=label); CntPlot += 1
	plt.xlim([0, 40000]); plt.ylim([0.65,0.8])
        if len(val_acc5):
            smoothed_val_acc5 = smooth(val_acc5[:,1], 10)
            myprint (diff(smoothed_val_acc5 * 100))
            plt.plot(val_acc5[:,0], smoothed_val_acc5, "-", color=ColorSet[CntPlot%len(ColorSet)], label=label.replace("acc", "acc5")); CntPlot += 1
            
def smooth(L, window = 50):
    # print ("call smooth, window = ", window)
    out = []
    num = len(L)
    for i in range(num):
        if i >= window:
            out.append(np.average(L[i-window: i])) 
        else:
            out.append(np.average(L[:i+1]))
    return np.array(out) if type(L) == type(np.array([0])) else out

def diff(L):
    out = []
    num = len(L)
    for i in range(num-1):
        out.append(float("%.3f" % (L[i+1] - L[i])))
    return np.array(out) if type(L) == type(np.array([0])) else out

def myprint(L):
    flag = 1 if str(L[0])[0] != '-' else -1
    for i in L:
        new_flag = 1 if str(i)[0] != '-' else -1
        if new_flag != flag:
            print ("\n%6.3f" % i, end=" ")
            flag = new_flag
        else:
            print ("%6.3f" % i, end=" ")
    print ("")

def plot_prune(log_file):
    lines = [l.strip() for l in open(log_file)]
    pruned_ratio = {}
    learn_speed  = []
    time_prune_finish = {}
    
    for i in range(len(lines)):
        layer_name = lines[i].split("  ")[0]
        if "pruned_ratio" in lines[i] and not("prune finish" in lines[i]):
            if layer_name in pruned_ratio.keys():
                k = 1; 
                while(not ("Step" in lines[i-k])): 
                    k = k + 1
                pruned_ratio[layer_name].append([int(lines[i-k].split(" ")[2]), float(lines[i].split("pruned_ratio:")[1].split("prune_ratio")[0].strip())]) ## [step, prune_ratio]
            else:    
                t = 1
                while(not ("Step" in lines[i-t])):
                    t = t + 1
                pruned_ratio[layer_name] = [[int(lines[i-t].split(" ")[2]), float(lines[i].split("pruned_ratio:")[1].split("prune_ratio")[0].strip())]]
                
        elif "learning_speed" in lines[i]:
            t = 1
            while(not ("Step" in lines[i-t])):
                t = t + 1
            learn_speed.append([int(lines[i-t].split(" ")[2]), float(lines[i].split(" ")[-1])])

        elif "prune finish" in lines[i].split("  ")[0]:
            layer_name = lines[i].split(" ")[0]            
            if layer_name in time_prune_finish.keys():
                k = 1
                while(not ("Step" in lines[i-k])): 
                    k = k + 1
                time_prune_finish[layer_name].append(int(lines[i-k].split(" ")[2])) 
            else:    
                t = 1
                while(not ("Step" in lines[i-t])):
                    t = t + 1
                time_prune_finish[layer_name] = [int(lines[i-t].split(" ")[2].split(":")[0])]
            
    np.save(log_file.split(".txt")[0]+".npy", pruned_ratio)

    

    ##plt.subplot(2,1,1)
    # Plot Pruning Number
    for layer, r in pruned_ratio.items():
        layer_index = int(layer.split("[")[1].split("]")[0])
        r = np.array(r)
        plt.plot(r[:,0], r[:,1], "-", color = ColorSet[layer_index], label = layer)
    
    # Plot Pruning Complete Time 
    for layer, iter in time_prune_finish.items():
        print (iter)
        assert len(iter) == 1
        plt.plot([iter[0], iter[0]], [0, 1], "-k")
        
        
    plt.xlabel("Step"); plt.ylabel("pruned ratio")
    plt.legend()
    plt.grid(True)
    
    ## Plot learning speed
    # plt.subplot(2,1,2)
    # learn_speed = [[i[0], max(i[1],0)] for i in learn_speed] ## only plot positive values
    # learn_speed = np.array(learn_speed[50:]) ## begin from 10, to discard some trash data
    # learn_speed_undersampled = learn_speed[::200, :] 
    # plt.plot(learn_speed[:,0], learn_speed[:,1], "-b")
    # plt.plot(learn_speed_undersampled[:,0], learn_speed_undersampled[:,1], "-k")
    # plt.xlabel("Step"); plt.ylabel("Learning Speed")
    # plt.grid(True)
    
    # output_path = log_file.split(".txt")[0] + ".png"
    # plt.savefig(output_path, dpi = 500)
    # plt.close()
    print ("Plot pruning - done!")
    
def plot_acc_prune(f1, f2):
    if "acc" in f2: 
        f1, f2 = f2, f1 ## f1: acc  f2: prune, they must be ".npy" file.
    val_acc = np.load(f1)
    ratio   = np.load(f2)
    print (ratio)
    plt.plot(val_acc[:,0], val_acc[:,1], "-g", label = "validation accuracy")
    for layer, r in ratio.items():
        layer_index = int(layer.split("[")[1].split("]")[0])
        r = np.array(r)
        plt.plot(r[:][0], r[:][1], "-", color = ColorSet[layer_index], label = layer)
    
    plt.xlabel("Step")
    plt.grid(True)
    plt.legend()
    plt.savefig(f1.split("_acc.npy")[0] + "_acc_prune.png", dpi = 500)
    print ("Plot pruning && acc - done!")
    

def compare_acc_trajectory(accFiles):
    for f in accFiles:
        timeID = f.split(os.sep)[-1].split("_")[1]
        if not os.path.exists(f):
            f = os.path.join(os.path.split(accFiles[0])[0], f) # if accFiles are in the same dir, only the first needs to provide dir path
        plot_acc(f, timeID)
    plt.grid(True)
    plt.legend()
    plt.xlabel("Step"); plt.ylabel("Accuracy and Loss")
    plt.savefig(f.split(".txt")[0] + ".png", dpi=200)
    plt.close()
    print ("Plot accuracy - done!")
    
if __name__ == "__main__":
    '''Usage:
        (1) compare acc or see if get acc plateau: 
            `python  this_file.py  **/weights/log_192-20180123-1608_retrain_acc.txt  **/weights/log_192-20180123-1632_retrain_acc.txt`
            `python  this_file.py  **/weights/log_192-20180123-1608_retrain_acc.txt             log_192-20180123-1632_retrain_acc.txt`
        (2) 
    '''
    compare_acc_trajectory(sys.argv[1:])
    # path = str(sys.argv[1]).split("/log")[0]
    # timeID = str(sys.argv[1]).split("/log")[1].split("_")[1]
    # print ("time stamp is: " + timeID)
    # files = [os.path.join(path, i) for i in os.listdir(path) if timeID in i and os.path.splitext(i)[1] == ".txt"]
    # files.sort() ## sort to make sure plot acc first

    # for f in files:
        # if "log" in f and "acc" in f:
            # plot_acc(f, timeID)
            # plt.grid(True)
            # plt.legend()
            # plt.xlabel("Step"); plt.ylabel("Accuracy and Loss")
            # plt.savefig(f.split(".txt")[0] + ".png", dpi = 500)
            # plt.close()
            # print ("Plot accuracy and loss - done!")
    
        # elif "log" in f and "prune" in f:
            # plot_prune(f)
            
        # else:
            # print ("Wrong! No function to deal with '%s'" % f)
    # files = [os.path.join(path, i) for i in os.listdir(path) if timeID in i and os.path.splitext(i)[1] == ".npy"]
    # if len(files) == 2:
        # plot_acc_prune(files[0], files[1])
    
