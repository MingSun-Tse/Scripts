from __future__ import print_function
'''Usage:
    This file is to visualize the trajectory of weight groups (like row or columns) during pruning.
    python  this_file.py  ***ave_magnitude.log
'''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import cPickle as pickle

Markers = { "conv1": ".", 
            "conv2": "x",
            "conv3": "d",
}
Colors = { "conv1": "g",
           "conv2": "r",
           "conv3": "b",
}

def main():
    assert (len(sys.argv) == 2)
    inFile = sys.argv[1]
    trajectory = {}
    cnt = 0
    for line in open(inFile):
        # cnt += 1
        # if cnt > 10000:
            # break
        line = line.strip()
        iter    = int(line.split(" ")[1])
        layer   = line.split(" ")[2][:-1]
        ave_mag = [float(i) for i in line.split(" ")[3:]]
        if layer in trajectory.keys():
            trajectory[layer].append([iter] + ave_mag)
        else:
            trajectory[layer] = [[iter] + ave_mag]

    # save
    # outPkl = inFile.replace(".log", ".pkl")
    # with open(outPkl, 'w') as f:
        # pickle.dump(trajectory, f)
    step = 10
    layers = sorted(trajectory.keys())
    for layer in layers:
        traj = np.array(trajectory[layer])
        num_col = traj.shape[1] - 1
        for i in np.random.permutation(num_col):
            plt.plot(traj[:,0][::step], traj[:,i+1][::step], color = Colors[layer], alpha=0.25, label = layer+"-"+str(i), linewidth = 0.2)
        plt.xlabel("Iteration"); plt.ylabel("Average magnitude of weight column")
        plt.savefig(inFile.replace(".log", "_" + layer + ".pdf"))
        plt.close()

if __name__ == "__main__":
    main()

