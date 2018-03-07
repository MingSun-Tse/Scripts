import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle

'''Usage:
    python  this_file.py  ***acc.txt
    python  this_file.py  ***/          # if directory provided, it will look for the only *acc.txt automatically.
'''
GFLOPs = [2457600, 6553600, 3276800]


# check arguments
assert(len(sys.argv) == 2)
inFile = sys.argv[1]
inFile = os.path.abspath(inFile)

# if directory provided
IF_dir = False
if os.path.isdir(inFile):
    IF_dir = True
    prune_txts = [os.path.join(inFile, i) for i in os.listdir(inFile) if i.endswith("prune.txt") and "retrain" not in i]
    if len(prune_txts) != 1:
        print ("there are not 1 `prune.txt` in your provided directory, please check.")
        exit(1)
    inFile = prune_txts[0]
output_path = sys.argv[1] if IF_dir else os.sep + os.path.join(*inFile.split(os.sep)[:-1])
timeID = inFile.split("log_")[1].split("_")[0]

def get_speedup(csp, rsp, GFLOPs):
    c_left = [1-i for i in csp]
    r_left = [1-i for i in rsp]
    assert(len(csp) == len(rsp) == len(GFLOPs))
    left_GFLOPs = [c_left[i] * r_left[i] * GFLOPs[i] for i in range(len(c_left))]
    return sum(GFLOPs) / sum(left_GFLOPs)

col_sparsity = []
row_sparsity = []
speedup = []
lines = [l.strip() for l in open(inFile).readlines()]
for i in range(len(lines)):
    if "Column Sparsity %" in lines[i]:
        k = 1
        while ("Iteration" not in lines[i-k]):
            k += 1
        iter = int(lines[i-k].split("Iteration ")[1].split(",")[0])
        csp = [float(sp)/100 for sp in lines[i+1].split("\t")[::2][:-1]]
        rsp = [float(sp)/100 for sp in lines[i+3].split("\t")[::2][:-1]]
        col_sparsity.append([iter] + csp) # ONLY work for ConvNet, for other nets, this probabaly needs changes.
        row_sparsity.append([iter] + rsp)
        speedup.append([iter, get_speedup(csp, rsp, GFLOPs)])
        print (csp, rsp, get_speedup(csp, rsp, GFLOPs))

col_sparsity = np.array(col_sparsity)
col_sparsity = np.array(row_sparsity)
print ("Plotting")
speedup = np.array(speedup)
plt.plot(col_sparsity[:,0], col_sparsity[:,1], label = "conv1")
plt.plot(col_sparsity[:,0], col_sparsity[:,2], label = "conv2")
plt.plot(col_sparsity[:,0], col_sparsity[:,3], label = "conv3")
plt.plot(speedup[:,0], speedup[:,1], label = "speedup")
plt.legend()
plt.savefig(inFile.replace("_acc.txt", "_speedup.pdf"))

    