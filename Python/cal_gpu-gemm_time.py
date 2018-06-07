from __future__ import print_function
import sys

'''Usage:
example of the log:
    I0301 11:58:35.465884 32183 base_conv_layer.cpp:833] conv1	 group 0: 16.64 us (Concatenation Timing)
    I0301 11:58:35.466332 32183 base_conv_layer.cpp:833] conv1	 group 0: 15.104 us (Concatenation Timing)
    I0301 11:58:35.466789 32183 base_conv_layer.cpp:833] conv1	 group 0: 14.56 us (Concatenation Timing)
    I0301 11:58:35.467243 32183 base_conv_layer.cpp:833] conv1	 group 0: 14.528 us (Concatenation Timing)

python  this_file.py  **log_file_like_above**  alexnet
'''
assert(len(sys.argv) == 3)
inFile   = sys.argv[1]
net_name = sys.argv[2]

# the number of rows and columns
alexnet = {
"conv1": [96,  363], # 96 rows and 363 columns
"conv2": [256, 1200],
"conv3": [384, 2304],
"conv4": [384, 1728],
"conv5": [384, 1728],
}

vgg16 = {
"conv1_1": [64, 27],
"conv1_2": [64, 576],
"conv2_1": [128, 576],
"conv2_2": [128, 1152],
"conv3_1": [256, 1152],
"conv3_2": [256, 2304],
"conv3_3": [256, 2304],
"conv4_1": [512, 2304],
"conv4_2": [512, 4608],
"conv4_3": [512, 4608],
"conv5_1": [512, 4608],
"conv5_2": [512, 4608],
"conv5_3": [512, 4608],
}

resnet50 = {
"conv1": [64, 147],
"res2a_branch1": [256, 64],
"res2a_branch2a": [64, 64],
"res2a_branch2b": [64, 576],
"res2a_branch2c": [256, 64],
"res2b_branch2a": [64, 256],
"res2b_branch2b": [64, 576],
"res2b_branch2c": [256, 64],
"res2c_branch2a": [64, 256],
"res2c_branch2b": [64, 576],
"res2c_branch2c": [256, 64],
"res3a_branch1": [512, 256],
"res3a_branch2a": [128, 256],
"res3a_branch2b": [128, 1152],
"res3a_branch2c": [512, 128],
"res3b_branch2a": [128, 512],
"res3b_branch2b": [128, 1152],
"res3b_branch2c": [512, 128],
"res3c_branch2a": [128, 512],
"res3c_branch2b": [128, 1152],
"res3c_branch2c": [512, 128],
"res3d_branch2a": [128, 512],
"res3d_branch2b": [128, 1152],
"res3d_branch2c": [512, 128],
"res4a_branch1": [1024, 512],
"res4a_branch2a": [256, 512],
"res4a_branch2b": [256, 2304],
"res4a_branch2c": [1024, 256],
"res4b_branch2a": [256, 1024],
"res4b_branch2b": [256, 2304],
"res4b_branch2c": [1024, 256],
"res4c_branch2a": [256, 1024],
"res4c_branch2b": [256, 2304],
"res4c_branch2c": [1024, 256],
"res4d_branch2a": [256, 1024],
"res4d_branch2b": [256, 2304],
"res4d_branch2c": [1024, 256],
"res4e_branch2a": [256, 1024],
"res4e_branch2b": [256, 2304],
"res4e_branch2c": [1024, 256],
"res4f_branch2a": [256, 1024],
"res4f_branch2b": [256, 2304],
"res4f_branch2c": [1024, 256],
"res5a_branch1": [2048, 1024],
"res5a_branch2a": [512, 1024],
"res5a_branch2b": [512, 4608],
"res5a_branch2c": [2048, 512],
"res5b_branch2a": [512, 2048],
"res5b_branch2b": [512, 4608],
"res5b_branch2c": [2048, 512],
"res5c_branch2a": [512, 2048],
"res5c_branch2b": [512, 4608],
"res5c_branch2c": [2048, 512],
}

NumRowCol = {
'alexnet': alexnet,
'caffenet': alexnet,
'vgg16': vgg16,
'resnet50': resnet50
}


gpu_gemm_time = {} # layer_name: [total_time, time_iter]
speedup = {}
matrix_left = {}
IF_after_Benchmark = False

for line in open(inFile):
    line = line.strip()
    if not IF_after_Benchmark:
        if "Benchmark begins" in line:
            IF_after_Benchmark = True # find the "Benchmark begins", don't count the timing before "Benchmark begins"
            
    if "squeezing to" in line:
        num_row_left = int(line.split("squeezing to ")[1].split("x")[0])
        num_col_left = int(line.split("squeezing to ")[1].split("x")[1])
        layer_name = line.split("]")[1].split(" ")[1]
        if layer_name in matrix_left.keys():
            matrix_left[layer_name] += num_col_left * num_row_left
        else:
            matrix_left[layer_name] = num_col_left * num_row_left;
        num_row = NumRowCol[net_name][layer_name][0]
        num_col = NumRowCol[net_name][layer_name][1]
        speedup[layer_name] = num_row * num_col * 1.0 / matrix_left[layer_name]
    
    if IF_after_Benchmark and line.lower().endswith("timing)"):
        layer_name = line.split("\t")[0].split(" ")[-1]
        if layer_name in gpu_gemm_time.keys():
            gpu_gemm_time[layer_name][0] += float(line.split("]")[1].split(" ")[4])
            if line.split("]")[1].split(" ")[3] == "0:":
                gpu_gemm_time[layer_name][1] += 1 # only count once for group > 1
        else:
            gpu_gemm_time[layer_name] = [float(line.split("]")[1].split(" ")[4]), 1]
            assert(line.split("]")[1].split(" ")[3] == "0:") # group should be `group: 0` because it's the first time meeting this layer

layers = list(gpu_gemm_time.keys())
layers.sort()
sum_ave_time = 0
for layer in layers:
    ave_time = gpu_gemm_time[layer][0] / gpu_gemm_time[layer][1]
    sum_ave_time += ave_time

for layer in layers:
    if layer not in speedup.keys():
        speedup[layer] = 1.0
    ave_time = gpu_gemm_time[layer][0] / gpu_gemm_time[layer][1]
    output = "{:25s}  {:10.1f} us  {:-.1f}%  ({} examples)  theoretical_speedup:{:.2f}"
    print(output.format(layer,
                        ave_time, 
                        ave_time * 100.0 / sum_ave_time,
                        gpu_gemm_time[layer][1],
                        speedup[layer]))

# TODO: sort by time percentage