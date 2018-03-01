from __future__ import print_function
import sys

'''Usage:
example of the log:
    I0301 11:58:35.465884 32183 base_conv_layer.cpp:833] conv1	 group 0: 16.64 us (Concatenation Timing)
    I0301 11:58:35.466332 32183 base_conv_layer.cpp:833] conv1	 group 0: 15.104 us (Concatenation Timing)
    I0301 11:58:35.466789 32183 base_conv_layer.cpp:833] conv1	 group 0: 14.56 us (Concatenation Timing)
    I0301 11:58:35.467243 32183 base_conv_layer.cpp:833] conv1	 group 0: 14.528 us (Concatenation Timing)

python  this_file.py  **log_file_like_above**
'''
assert(len(sys.argv) == 2)
inFile = sys.argv[1]

gpu_gemm_time = {}
for line in open(inFile):
    line = line.strip()
    if not line.endswith("Timing)"): continue
    layer_name = line.split("\t")[0].split(" ")[-1]
    if layer_name in gpu_gemm_time.keys():
        gpu_gemm_time[layer_name][0] += float(line.split(" ")[7])
        gpu_gemm_time[layer_name][1] += 1
    else:
        gpu_gemm_time[layer_name] = [float(line.split(" ")[7]), 1]

layers = list(gpu_gemm_time.keys())
layers.sort()
for layer in layers:
    ave_time = gpu_gemm_time[layer][0] / gpu_gemm_time[layer][1]
    print("{}: {} us ({} examples)".format(layer, ave_time, gpu_gemm_time[layer][1]))
