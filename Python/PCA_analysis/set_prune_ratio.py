import os
import sys

"""Usage:
    python  this_file  model.prototxt  log_192-20180418-0138_finish_pruned_ratio.wh # get prune_ratio from pre-trained log
	python  this_file  model.prototxt  pruning_ratio_result_speedup_2.txt # get prune_ratio from PCA analysis result
"""
assert(len(sys.argv) == 3)
model = sys.argv[1]
inFile = sys.argv[2]

# Find pruned ratio from pre-trained log
pruned_ratio = {}
for line in open(inFile):
    if "prune finished!" in line:
        layer = line.split(" ")[0]
        ratio = float(line.split("pruned_ratio_col:")[1].split("  ")[0].strip()) # Note that, only column pruning needs this way to set prune_ratio
        assert(layer not in pruned_ratio.keys())
        pruned_ratio[layer] = ratio

# Find prune ratio from PCA
for line in open(inFile):
    if "pruning ratio of" in line:
		layer = line.split("of")[1].split(":")[0].strip()
		ratio = float(line.split(":")[1].split("(")[0].strip())
		assert(layer not in pruned_ratio.keys())
		pruned_ratio[layer] = ratio

# Set prune_ratio in prototxt
out_model_prototxt = open(model.replace(".prototxt", "_prune_ratio_set_done.prototxt"), "w+")
lines = [i for i in open(model)]
for i in range(len(lines)):
    new_line = lines[i]
    if "prune_ratio_step" in lines[i]:
        k = 1
        while "name" not in lines[i-k]:
            k += 1
        layer = lines[i-k].split('"')[1]
        if layer in pruned_ratio.keys():
            new_line = lines[i].split("prune_ratio_step")[0] + "prune_ratio_step: " + str(pruned_ratio[layer]) + "\n"
    out_model_prototxt.write(new_line)
out_model_prototxt.close()
print("set prune ratio done")
