# coding: utf-8
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import sys


def sum_(x):
    result = []
    for i in range(len(x)):
        result.append(sum(x[:i+1]))
    return result
    
def pca_analysis(w, keep_ratio, IF_fig=False, layer=None):
    w = np.array(w)
    num_row, num_col = w.shape[0], w.shape[1]
    n_components = int(np.ceil(keep_ratio * num_col))
    pca = PCA(n_components=n_components)

    # substract mean
    mean = np.average(w, axis = 0)
    #w = w - mean
    
    # results
    coff  = pca.fit_transform(w)
    eigen = pca.components_
    #print (eigen)
    #print (pca.explained_variance_ratio_.sum())
    w_    = coff.dot(eigen)

    # 原数据点的模平方
    sum = 0
    for i in range(num_row):
        sum += (np.dot(w[i], w[i]))
    sum /= num_row

    # 误差的模平方
    err   = (w - w_)
    sum_e = 0
    for i in range(num_row):
        sum_e += (np.dot(err[i], err[i]))
    sum_e /= num_row
    
    if IF_fig:
        assert layer != None
        y = pca.explained_variance_
        plt.plot(y, 'k-.')
        plt.plot(sum_(y), 'r-')
        plt.xlabel("# Eigen Vector")
        plt.ylabel("Explained Veriance")
        plt.savefig(os.path.join(sys.argv[1], layer+".pdf"))
        plt.close()
    
    # print (keep_ratio, sum_e / sum)
    return sum_e / sum

    
# Some settings
Layers = {
"vgg16": ["conv1_1", "conv1_2", 
          "conv2_1", "conv2_2", 
          "conv3_1", "conv3_2", "conv3_3", 
          "conv4_1", "conv4_2", "conv4_3", 
          "conv5_1", "conv5_2", "conv5_3"],
                      
"alexnet": ["conv1", "conv2", "conv3", "conv4", "conv5"],
           
"resnet50": ["conv1",

            "res2a_branch2b",
            "res2b_branch2b",
            "res2c_branch2b",

            "res3a_branch2b",
            "res3b_branch2b", 
            "res3c_branch2b", 
            "res3d_branch2b", 

            "res4a_branch2b", 
            "res4c_branch2b", 
            "res4e_branch2b", 
            "res4f_branch2b",  
            
            "res5a_branch2b", 
            "res5b_branch2b", 
            "res5c_branch2b"],
            
"inceptionv3": ["conv1_3x3_s2",
                "inception_a1_3x3_1",
                "inception_b2_7x1_3"],
}
# "resnet50": ["conv1",

            # "res2a_branch1", 
            # "res2a_branch2a", "res2a_branch2b", "res2a_branch2c",
            # "res2b_branch2a", "res2b_branch2b", "res2a_branch2c",
            # "res2c_branch2a", "res2c_branch2b", "res2c_branch2c",
            
            # "res3a_branch1", 
            # "res3a_branch2a", "res3a_branch2b", "res3a_branch2c",
            # "res3b_branch2a", "res3b_branch2b", "res3b_branch2c",
            # "res3c_branch2a", "res3c_branch2b", "res3c_branch2c",
            # "res3d_branch2a", "res3d_branch2b", "res3d_branch2c",
            
            # "res4a_branch1",
            # "res4a_branch2a", "res4a_branch2b", "res4a_branch2c",
            # "res4b_branch2a", "res4b_branch2b", "res4b_branch2c",
            # "res4c_branch2a", "res4c_branch2b", "res4c_branch2c",
            # "res4d_branch2a", "res4d_branch2b", "res4d_branch2c",               
            # "res4e_branch2a", "res4e_branch2b", "res4e_branch2c",
            # "res4f_branch2a", "res4f_branch2b", "res4f_branch2c",      
            
            # "res5a_branch1",
            # "res5a_branch2a", "res5a_branch2b", "res5a_branch2c",
            # "res5b_branch2a", "res5b_branch2b", "res5b_branch2c",
            # "res5c_branch2a", "res5c_branch2b", "res5c_branch2c",
           
color = {'conv1': 'r', 
         'conv2': 'b', 
         'conv3': 'g', 
         'conv4': 'k', 
         'conv5': 'y',
         
         'res2' : 'b',
         'res3' : 'g',
         'res4' : 'k',
         'res5' : 'y',

        }

marker = {'conv1': 'd', 
          'conv2': 'o', 
          'conv3': '*', 
          'conv4': 'x', 
          'conv5': 's',
          
          'res2' : 'o',
          'res3' : '*',
          'res4' : 'x',
          'res5' : 's',

         }
         
linestyle = ['solid', 'dashed', 'dotted', '-.']

def main(dir, what_net):
    err_ratio = {}
    max_  = 1
    step_ = 0.05
    min_  = 0
    layers = Layers[what_net]
    for layer in layers:
        w = os.path.join(dir, "weights_" + layer + ".npy")
        err_ratio[layer] = []
        for keep_ratio in np.arange(min_, max_, step_):
            IF_fig = True if keep_ratio == max_ - step_ else False
            l = layer if IF_fig else None
            err_ratio[layer].append(pca_analysis(np.load(w).T, keep_ratio, IF_fig, l))
        print ("%s PCA done" % layer)

    # Plot
    for layer in layers:
        x = np.arange(min_, max_, step_) * 100
        y = np.array(err_ratio[layer]) * 100
        stage = layer.split('_')[0][:4] if what_net == 'resnet50' else layer.split('_')[0] # stage of different layers
        stage = "conv1" if layer == "conv1" else stage # for resnet50        
        index = ord(layer[4])-ord('a')  if what_net == 'resnet50' else int(layer.split('_')[1])-1 if len(layer.split('_')) > 1 else 0 # index in each stage
        
        # handly set the stage for resnet50, quite fussy
        if layer == "res4c_branch2b":
            index = 1
        if layer == "res4e_branch2b":
            index = 2
        if layer == "res4f_branch2b":
            index = 3 
            
        plt.plot(x, y, label=layer, linewidth=1, color=color[stage], 
                    marker=marker[stage], markersize=4 , linestyle=linestyle[index%4])
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.ylabel("Normalized Reconstruction Error ($\%$)", size = 12)
    plt.xlabel("Remaining Dimension ($\%$)", size = 12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_sensitivity.pdf" % what_net)
    #plt.show()

def main_inceptionv3():

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
