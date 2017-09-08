import sys
def add_param(protofile, prune_ratios, deltas):
    '''
        Usage: python convert_prototxt.py  alexnet.prototxt  /0~5:0.75,6-8:0.5,9:0.4/  /:0.1/
        Do NOT use blanks in /.../
        0~5 -- layer 0 to layer 5
        6-8 -- layer 6 and layer 8
        9   -- layer 9
        :   -- delta, if layer_index before ':' missing, in default, set the delta of all the layers with prune ratio to 0.1
        
    '''
    assert protofile.split(".")[-1] == "prototxt"
    out = open(protofile.replace(".prototxt", "_added.prototxt"), "w+")
    lines = [l for l in open(protofile, "r")]
    
    # parsing to get prune_ratio of every layer
    layer_pratio = {}
    layers = prune_ratios.split("/")[1].split(",")
    for l in layers:
        if "~" in l.split(":")[0]:
            begin = int(l.split(":")[0].split("~")[0])
            end   = int(l.split(":")[0].split("~")[1])
            for i in range(begin, end+1):
                if str(i) in layer_pratio.keys(): 
                    print "Error: set prune_ratio of layer %s more than once." % i
                    return
                layer_pratio[str(i)] = l.split(":")[1]
                
        elif "-" in l.split(":")[0]:
            for i in l.split(":")[0].split("-"):
                if i in layer_pratio.keys(): 
                    print "Error: set prune_ratio of layer %s more than once." % i
                    return
                layer_pratio[i] = l.split(":")[1]
        
        elif l.split(":")[0].isdigit():
            if l.split(":")[0] in layer_pratio.keys(): 
                print "Error: set prune_ratio of layer %s more than once." % l.split(":")[0]
                return
            layer_pratio[l.split(":")[0]] = l.split(":")[1]
        
        else:
            print "Error: wrong format of prune_ratio, which should be ONLY like {1~5:0.75, 6-8:0.5, 9:0.4}."
            return

    # add PruneParameter
    for i in range(len(lines)):
        if "convolution_param" in lines[i]:
            # serach up for name, get layer_index from name
            t = 1 
            while lines[i-t].strip()[:4] != "name" and lines[i-t].strip()[:4] != "layer":
                t += 1
            if lines[i-t].strip()[:4] == "name":
                layer_index = lines[i-t].split('"')[1].split("[")[1].split("]")[0]
            else: # search down
                t = 1
                while lines[i+t].strip()[:4] != "name":
                    t += 1
                layer_index = lines[i+t].split('"')[1].split("[")[1].split("]")[0]
            
            # got layer_index
            delta = '0'
            if layer_index in layer_pratio.keys():
                pratio = layer_pratio[layer_index]
                # add prune_ratio just before convolution_param{...}
                blanks = lines[i].split("convolution_param")[0]
                out.write(blanks + "prune_param {\n" 
                          + blanks*2 + "prune_ratio: " + pratio + "\n"
                          + blanks*2 + "delta: " + delta + "\n" 
                          + blanks + "}\n")
            
        out.write(lines[i])
    out.close()
    # print and check
    ks = [int(k) for k in layer_pratio.keys()]
    print "#" + "  " + "prune_ratio" + "  " + "delta" 
    for k in sorted(ks):
        print str(k) + "  " + layer_pratio[str(k)] + "  " + delta

def rename_layer(protofile):
    '''
        Another idea: serach "Convolution", then replace the name, this may be simpler.
    '''
    assert protofile.split(".")[-1] == "prototxt"
    out = open(protofile.replace(".prototxt", "_renamed.prototxt"), "w+")
    lines = [l for l in open(protofile, "r")]
    layer_index = 0 # TODO: maybe we should allow for fc layers
    layer_names = []
    
    
    IF_meet_first_layer = False
    for i in range(len(lines)):
        # before we meet the first layer, just move down
        if not IF_meet_first_layer: 
            out.write(lines[i])
            if lines[i].strip()[:5] == "layer":
                IF_meet_first_layer = True
            continue
        if lines[i].strip()[:4] == "name":
            # serach down for type
            t = 1 
            while not lines[i+t].strip()[:5] in ("type:", "layer"):  
                t += 1
            if lines[i+t].strip()[:5] == "type:":
                type = lines[i+t].split('"')[1]
            # search up for type
            else: 
                t = 1
                while lines[i-t].strip()[:5] != "type:":
                    t += 1
                type = lines[i+t].split('"')[1]
            # after finding the type 
            if type == "Convolution": # If there is a name and this name is of conv layer, then replace it, otherwise just keep it was.
                newl = lines[i].replace(lines[i].split('"')[1], "[" + str(layer_index) + "]" + lines[i].split('"')[1])
                layer_names.append("[" + str(layer_index) + "]" + lines[i].split('"')[1])
                layer_index += 1
            else: 
                newl = lines[i]
        else:
            newl = lines[i]
        out.write(newl)
    out.close()
    for i in layer_names: print i
        
    
    

if __name__ == "__main__":
    if len(sys.argv) == 2:
        rename_layer(str(sys.argv[1]))
    elif len(sys.argv) == 4:
        add_param(*sys.argv[1:])
    