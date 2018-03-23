import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle

'''Usage
python  this_file.py  ***prune.txt

'''
assert(len(sys.argv) == 2)
inFile = sys.argv[1]
inFile = os.path.abspath(inFile)

# cat
pruned_ratio_log_file = inFile.replace("_prune.txt", "_pruned_ratio.log")
output_path = os.sep + os.path.join(*inFile.split(os.sep)[:-1])
script = "cat " + inFile + "| grep 'IF_prune' | grep 'conv' > " + pruned_ratio_log_file
print (script)
os.system(script)


layers = {}
for line in open(pruned_ratio_log_file):
    layer = line.split("  ")[0]
    pruned_ratio     = float(line.split("  ")[2].split(" ")[1])
    pruned_ratio_row = float(line.split("  ")[3].split(" ")[1].split("(")[0])
    pruned_ratio_col = float(line.split("  ")[4].split(" ")[1].split("(")[0])
    
    if layer in layers.keys():
        layers[layer].append([pruned_ratio, pruned_ratio_row, pruned_ratio_col])
    else:
        layers[layer] = [[pruned_ratio, pruned_ratio_row, pruned_ratio_col]]

# save for future use
# output_pkl = pruned_ratio_log_file.replace(".log", ".pkl")
# with open(output_pkl, 'w') as f:
    # pickle.dumps(layers, f)
for layer in layers.keys():
    np.save(pruned_ratio_log_file.replace(".log", "_"+layer+".npy"), layers[layer])

    
items = ["pruned_ratio"]
layers_ = sorted(layers.keys())
for ix in range(len(items)):
    for layer in layers_:
        p = np.array(layers[layer])
        plt.plot(p[:,ix], label = items[ix]+"-"+layer)

plt.title("pruned_ratio of different layers during pruning")
plt.legend()
plt.savefig(pruned_ratio_log_file.replace(".log", ".pdf"))
plt.close()