import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

color_set = ("r", "b", "g", "k", "c", "y")
nets = ("caffenet", "cifar10_full")
net = "None"
col_num = {"cifar10_full":(75, 800, 800), 
           "caffenet":(363, 1200, 2304, 1728, 1728)} # layer 2,4,5, group = 2

def plot_acc(log_file):
    val_acc = []; val_acc_top_5 = []; val_loss = []
    train_loss= []; train_loss_smoothed  = []
    
    lines = [l.strip() for l in open(log_file)]
    IF_find_net = False
    for i in range(len(lines)):
        if ~IF_find_net and '"' in lines[i] and lines[i].lower().split('"')[1] in nets: 
            global net
            net = lines[i].lower().split('"')[1]
            IF_find_net = True
            
    
        if "Test net output" in lines[i] and "accuracy = " in lines[i]: # val acc
            j = 1
            while(not lines[i-j].split("Iteration ")[-1].split(",")[0].isdigit()):
                j = j + 1
            val_acc.append([int(lines[i-j].split("Iteration ")[-1].split(",")[0]), float(lines[i].split("= ")[-1])])

        if "Test net output" in lines[i] and "accuracy_top_5 = " in lines[i]: # val acc top-5 
            j = 1
            while(not lines[i-j].split("Iteration ")[-1].split(",")[0].isdigit()):
                j = j + 1
            val_acc_top_5.append([int(lines[i-j].split("Iteration ")[-1].split(",")[0]), float(lines[i].split("= ")[-1])])
            
        if "Test net output" in lines[i] and "loss" in lines[i]: # val loss
            k = 2
            while(not lines[i-k].split("Iteration ")[-1].split(",")[0].isdigit()):
                k = k + 1
            val_loss.append([int(lines[i-k].split("Iteration ")[-1].split(",")[0]), float(lines[i].split("loss = ")[-1].split(" ")[0])])
        
        if "Train net output" in lines[i] and "loss" in lines[i]: # train loss
            t = 1
            while(not lines[i-t].split("Iteration ")[-1].split(",")[0].isdigit()):
                t = t + 1
            train_loss.append([int(lines[i-t].split("Iteration ")[-1].split(",")[0]), float(lines[i].split("loss = ")[-1].split(" ")[0])]) 
        
        if "Iteration" in lines[i] and "loss = " in lines[i]: # smoothed train loss
            train_loss_smoothed.append([int(lines[i].split("Iteration ")[-1].split(",")[0]), float(lines[i].split("loss = ")[-1])])

    assert(len(val_acc) == len(val_loss))
    train_loss = np.array(train_loss)
    train_loss_smoothed = np.array(train_loss_smoothed)
    val_loss = np.array(val_loss)
    val_acc = np.array(val_acc)
    val_acc_top_5 = np.array(val_acc_top_5)
    
    plt.subplot(211)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.plot(train_loss[:,0], train_loss[:,1], "-b", label = "train loss")
    plt.plot(val_loss[:,0], val_loss[:,1], "-r", label = "validation loss")
    plt.plot(train_loss_smoothed[:,0], train_loss_smoothed[:,1], "-k", label = "smoothed train loss")
    plt.grid(True)
    plt.legend()
    
    
    plt.subplot(212)
    plt.xlabel("Step"); plt.ylabel("Accuracy")
    plt.plot(val_acc[:,0], val_acc[:,1], "-g", label = "validation")
    if len(val_acc_top_5) != 0:
        plt.plot(val_acc_top_5[:,0], val_acc_top_5[:,1], "-c", label = "validation top-5")
    plt.plot(val_acc[:,0], [0.81] * len(val_acc[:,0]), "-r", label = "acc = 0.81")
    plt.grid(True)
    plt.legend()
    
    #Save 
    plt.savefig(log_file.split(".txt")[0] + ".png", dpi = 500)
    plt.close()
    np.save(log_file.split(".txt")[0] + ".npy", val_acc)
    print ("Plot accuracy and loss - done!")
    

def plot_prune(log_file):
    lines = [l.strip() for l in open(log_file)]
    prune_num = {}
    lspeed = []
    prune_complete_time = {}
    update_prob_time = {}
    recover_prob_time = {}
    
    for i in range(len(lines)):
        
        if lines[i][26:43] == "num_pruned_column":
            if lines[i][:5] in prune_num.keys():
                k = 1; 
                while(not ("Step" in lines[i-k])): 
                    k = k + 1
                prune_num[lines[i][:5]].append([int(lines[i-k].split(" ")[2]), int(lines[i].split("num_pruned_column:")[1].split("num_col_to_prune")[0].strip())]) # [step, num_pruned_column]
            else:    
                t = 1
                while(not ("Step" in lines[i-t])):
                    t = t + 1
                prune_num[lines[i][:5]] = [[int(lines[i-t].split(" ")[2]), int(lines[i].split("num_pruned_column:")[1].split("num_col_to_prune")[0].strip())]]
                
        elif lines[i][:14] == "learning_speed":
            t = 1
            while(not ("Step" in lines[i-t])):
                t = t + 1
            lspeed.append([int(lines[i-t].split(" ")[2]), float(lines[i].split(" ")[-1])])
            
        elif lines[i][:6] == "update":
            if lines[i].split(" ")[-3] in update_prob_time.keys():
                update_prob_time[lines[i].split(" ")[-3]].append(int(lines[i].split(" ")[-1])) 
            else:    
                update_prob_time[lines[i].split(" ")[-3]] = [int(lines[i].split(" ")[-1])]
                
        elif lines[i][:7] == "recover":
            if lines[i].split(" ")[-3] in recover_prob_time.keys():
                recover_prob_time[lines[i].split(" ")[-3]].append(int(lines[i].split(" ")[-1])) 
            else:
                recover_prob_time[lines[i].split(" ")[-3]] = [int(lines[i].split(" ")[-1])]

        elif lines[i][:3] == "last":
            if lines[i][:5] in prune_complete_time.keys():
                k = 1
                while(not ("Step" in lines[i-k])): 
                    k = k + 1
                prune_complete_time[lines[i][:5]].append(int(lines[i-k].split(" ")[2])) 
            else:    
                t = 1
                while(not ("Step" in lines[i-t])):
                    t = t + 1
                prune_complete_time[lines[i][:5]] = [int(lines[i-t].split(" ")[2])]
            
            
    
    
    # Calcute pruning ratio and save
    prune_ratio = {}
    for layer, num in prune_num.items():
        num = np.array(num)
        prune_ratio[layer] = num[:,1] * 1.0 / col_num[net][int(layer[4]) - 1] # TODO: replace layer[4]
    ratios = np.array([num[:,0], prune_ratio]) 
    np.save(log_file.split(".txt")[0]+".npy", ratios)
    

    plt.subplot(2,1,1)
    # Plot Pruning Number
    for layer, num in prune_num.items():
        num = np.array(num)
        plt.plot(num[:,0], num[:,1], "-", color = color_set[int(layer[4])-1], label = layer)
    
    # Plot Pruning Complete Time 
    for layer, iter in prune_complete_time.items():
        print iter
        assert len(iter) == 1
        plt.plot([iter[0], iter[0]], [0, max(total_num.values())], "-k")
        
        
    plt.xlabel("Step"); plt.ylabel("Number of Pruned Column")
    plt.legend()
    plt.grid(True)
    
    # Plot learning speed
    plt.subplot(2,1,2)
    lspeed = [[i[0], max(i[1],0)] for i in lspeed] # only plot positive values
    lspeed = np.array(lspeed[50:]) # begin from 10, to discard some trash data
    lspeed_undersampled = lspeed[::200,:] 
    plt.plot(lspeed[:,0], lspeed[:,1], "-b")
    plt.plot(lspeed_undersampled[:,0], lspeed_undersampled[:,1], "-k")
    plt.xlabel("Step"); plt.ylabel("Learning Speed")
    plt.grid(True)
    
    output_path = log_file.split(".txt")[0] + ".png"
    plt.savefig(output_path, dpi = 500)
    plt.close()
    print ("Plot pruning - done!")
    
def plot_acc_prune(f1, f2):
    # RelativeRatio = {"conv1":0.2, "conv2":0.4, "conv3":0.4}
    if "acc" in f2: 
        f1, f2 = f2, f1 # f1: acc  f2: prune, they must be ".npy" file.
    val_acc = np.load(f1)
    ratios  = np.load(f2)
    plt.plot(val_acc[:,0], val_acc[:,1], "-g", label = "validation accuracy")
    for l, r in ratios[1].items():
        plt.plot(ratios[0], r, "-", color = color_set[int(l[4])-1], label = l)
    total_ratio = ratios[1]["conv1"] * 0.2 + ratios[1]["conv2"] * 0.4 + ratios[1]["conv3"] * 0.4
    plt.plot(ratios[0], total_ratio, "-k", label = "total pruning ratio")
    
    plt.xlabel("Step")
    plt.grid(True)
    plt.legend()
    plt.savefig(f1.split("_acc.npy")[0] + "_acc_prune.png", dpi = 500)
    print ("Plot pruning && acc - done!")
    
    
if __name__ == "__main__":
    assert len(sys.argv) == 2
    path = str(sys.argv[1]).split("/log")[0]
    timestamp = str(sys.argv[1]).split("/log")[1].split("_")[1]
    print ("Timestamp is: " + timestamp)
    files = [os.path.join(path, i) for i in os.listdir(path) if timestamp in i and os.path.splitext(i)[1] == ".txt"]
    files.sort() # sort to make sure plot acc first
    for f in files: # files must use complete path! 
        if "log" in f and "acc" in f:
            plot_acc(f)
        elif "log" in f and "prune" in f:
            plot_prune(f)
        else:
            print ("Wrong! No function to deal with '%s'" % f)
    files = [os.path.join(path, i) for i in os.listdir(path) if timestamp in i and os.path.splitext(i)[1] == ".npy"]
    if len(files) == 2:
        plot_acc_prune(files[0], files[1])
    
