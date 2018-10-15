from __future__ import print_function
'''Usage:
    This file is to visualize the trajectory of weight groups (like row or columns) during pruning.
    python  this_file.py  ***ave_magnitude.log
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

Markers = { 
"conv1": ".", 
"conv2": "x",
"conv3": "d",
}
Colors = { 
"conv1": "g",
"conv2": "r",
"conv3": "b",
}
PruneRatios = {
"conv1": 0.75, 
"conv2": 0.75,
"conv3": 0.75,
}

title_fs = 25
label_fs = 23
tick_label_fs = 20

def main():
    assert (len(sys.argv) == 3)
    inFile_SSL = sys.argv[1]
    inFile_ID  = sys.argv[2]

    for kk in range(2):
        trajectory = {}
        if kk == 0:
            inFile = inFile_SSL
            fig_title = "Fixed regularization"
        else:
            inFile = inFile_ID
            fig_title = "Varying regularization"
       

        print ("now parse the log file:\n  `%s`" % inFile)
        cnt = 0
        for line in open(inFile):
            line = line.strip()
            iter    = int(line.split(" ")[1])
            layer   = line.split(" ")[2][:-1]
            ave_mag = [float(i) for i in line.split(" ")[3:]]
            if layer in trajectory.keys():
                trajectory[layer].append([iter] + ave_mag)
            else:
                trajectory[layer] = [[iter] + ave_mag]
        print ("going to plot")
    
        # plot
        step = 1000
        layer = "conv1"
        traj = np.array(trajectory[layer])
        num_col = traj.shape[1] - 1
        
        sorted_col_score = sorted(traj[0, 1:])
        cols_to_prune = np.argsort(traj[0, 1:])[:int(np.ceil(PruneRatios[layer] * num_col))] # the columns which should be pruned based on their L1, at iter 0
        
        num_show_iter = 60000
        plt.subplot(1,2, kk+1)
        for i in range(num_col):
            plt.plot(traj[:num_show_iter,0][::step], traj[:num_show_iter,i+1][::step] * 32, color='k', alpha=1, label = layer+"-"+str(i), linewidth = 0.5)
            print ("Plotting %s-%s" % (layer, i))
        plt.xlabel("Iteration", fontsize=label_fs); plt.ylabel("$L_1$ norm", fontsize=label_fs)
        plt.xlim([0, num_show_iter + 10]); plt.ylim([0, 6])
        plt.xticks(fontsize=tick_label_fs); plt.yticks(fontsize=tick_label_fs)
        plt.title(fig_title, fontsize=title_fs)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

