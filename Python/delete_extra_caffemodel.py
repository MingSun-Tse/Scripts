'''Usage:
The saved caffemodel/solverstate should be at the interval of 1000. Namely, a caffemodel is saved every 1000 iterations. Any iterations which as not multiples of 1000 won't be deleted naturally.

argvs:
	argv 1, <dir>, is the directory to clean.
	<iter_not_del> are the iterations which won't be deleted.
	argv -1, bool value, is to indicate whether to delete all the other iterations except the ones in <iter_not_del>

Examples:
	python  this_file.py  <dir>  2000 150000 0
	python  this_file.py  <dir>  2000 150000 1
'''
import os
import sys
Dir = sys.argv[1]
assert (sys.argv[-1] in ('0', '1')) # the last argv is to indicate IF_delete_all_others, must be set
iter_not_del = sys.argv[2:-1]
IF_delete_all_others = int(sys.argv[-1])

for i in os.listdir(Dir):
    if ("caffemodel" in i or "solverstate" in i):
        iter = int(i.split("iter_")[1].split(".")[0])
        iter_interval = 10000000 if IF_delete_all_others else 5000
        if iter % 1000 == 0 and iter % iter_interval != 0 and str(iter) not in iter_not_del:
            os.remove(os.path.join(Dir, i))
print("delete done")
