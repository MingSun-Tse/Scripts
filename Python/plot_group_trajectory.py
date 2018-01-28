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

def main():
    assert (len(sys.argv) == 2)
    inFile = sys.argv[1]
    trajectory = {}

    outPKL = inFile.replace(".log", ".pkl")
    if os.path.exists(outPKL):
        print ("Loading existing pkl:\n  `%s`" % outPKL)
        trajectory = pickle.load(open(outPKL, 'r'))
        print ("Load done")
    else:
        print ("Not find existing pkl, now parse the log file:\n  `%s`" % inFile)
        for line in open(inFile):
            line = line.strip()
            iter    = int(line.split(" ")[1])
            layer   = line.split(" ")[2][:-1]
            ave_mag = [float(i) for i in line.split(" ")[3:]]
            if layer in trajectory.keys():
                trajectory[layer].append([iter] + ave_mag)
            else:
                trajectory[layer] = [[iter] + ave_mag]
        print ("Parsing done, saving pkl")
        # save
        with open(outPKL, 'w') as f:
            pickle.dump(trajectory, f)
        print ("Saving done, going to plot")
    
    # plot
    step = 500
    layers = sorted(trajectory.keys())
    for layer in layers:
        traj = np.array(trajectory[layer])
        num_col = traj.shape[1] - 1
        
        sorted_col_score = sorted(traj[0, 1:])
        cols_to_prune = np.argsort(traj[0, 1:])[:int(np.ceil(PruneRatios[layer] * num_col))] # the columns which should be pruned based on their L1, at iter 0
        
        for i in range(num_col):#np.random.permutation(num_col): # i: column number
            color = 'r' if i in cols_to_prune else 'b'
            plt.plot(traj[:,0][::step], traj[:,i+1][::step], color = color, alpha=0.25, label = layer+"-"+str(i), linewidth = 0.2)
            print ("Plotting %s-%s" % (layer, i))
        plt.xlabel("Iteration"); plt.ylabel("Average magnitude of weight column")
        plt.savefig(inFile.replace(".log", "_" + layer + ".pdf"))
        plt.close()

if __name__ == "__main__":
    main()

