import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

'''Usage
python  this_file.py  pruned_ratio.log

'''

assert(len(sys.argv) == 2)
inFile = sys.argv[1]
inFile = os.path.abspath(inFile)
layers = {}
for line in open(inFile):
    layer = line.split("  ")[0]
    pruned_ratio     = float(line.split("  ")[2].split(" ")[1])
    pruned_ratio_row = float(line.split("  ")[3].split(" ")[1].split("(")[0])
    pruned_ratio_col = float(line.split("  ")[4].split(" ")[1].split("(")[0])
    
    if layer in layers.keys():
        layers[layer].append([pruned_ratio, pruned_ratio_row, pruned_ratio_col])
    else:
        layers[layer] = [[pruned_ratio, pruned_ratio_row, pruned_ratio_col]]

items = ["pruned_ratio"]
layers_ = sorted(layers.keys())
for ix in range(len(items)):
    for layer in layers_:
        p = np.array(layers[layer])
        plt.plot(p[:,ix], label = items[ix]+"-"+layer)

plt.title("pruned_ratio of different layers during pruning")
plt.legend()
output_path = os.sep + os.path.join(*inFile.split(os.sep)[:-1])
print(output_path)
plt.savefig(os.path.join(output_path, "prund_ratio.pdf"))