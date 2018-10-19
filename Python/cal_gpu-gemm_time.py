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

# the number of rows and columns
lenet5 = {
"conv1": [20, 75],
"conv2": [50, 500],
}

alexnet = {
"conv1": [96,  363], # 96 rows and 363 columns
"conv2": [256, 1200],
"conv3": [384, 2304],
"conv4": [384, 1728],
"conv5": [384, 1728],
}

vgg16_cp4x = {
"conv1_1": [12, 27],
"conv1_2": [64, 108],
"conv2_1": [21, 576],
"conv2_2": [128, 189],
"conv3_1": [73, 1152],
"conv3_2": [58, 657],
"conv3_3": [256, 522],
"conv4_1": [121, 2304],
"conv4_2": [166, 1089],
"conv4_3": [512, 1494],
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
'lenet': lenet5,
'alexnet': alexnet,
'caffenet': alexnet,
'vgg_ilsvrc_16_layers': vgg16_cp4x,
'resnet-50': resnet50,
}


gpu_gemm_time = {}
speedup = {}
matrix_left = {}
net_name = ""
IF_find_net = False


for line in open(inFile):
    line = line.strip()
    if not IF_find_net:
        line = line.lower()
        if "name" in line and '"' in line: 
            net_name = line.split('"')[1].lower() # net name is using lower case
            print("the net is {}".format(net_name))
            IF_find_net = True
       
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
        
    if line.endswith("Timing)"):
        layer_name = line.split("\t")[0].split(" ")[-1]
        if layer_name in gpu_gemm_time.keys():
            gpu_gemm_time[layer_name][0] += float(line.split("]")[1].split(" ")[4])
            if line.split("]")[1].split(" ")[3] == "0:":
                gpu_gemm_time[layer_name][1] += 1
        else:
            gpu_gemm_time[layer_name] = [float(line.split("]")[1].split(" ")[4]), 1]
            assert(line.split("]")[1].split(" ")[3] == "0:") # group should be `group: 0`

layers = list(gpu_gemm_time.keys())
layers.sort()
total_conv_time = 0
for layer in layers:
    if layer not in speedup.keys():
        speedup[layer] = 1.0
    ave_time = gpu_gemm_time[layer][0] / gpu_gemm_time[layer][1]
    output = "{}: {} us ({} examples)  theoretical_speedup:{:.2f}"
    print(output.format(layer, ave_time, gpu_gemm_time[layer][1], speedup[layer]))
    total_conv_time += ave_time
print("total conv time: %.2f" % total_conv_time)
