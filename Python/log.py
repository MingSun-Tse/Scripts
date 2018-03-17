#!/usr/bin/env python
'''Usage:
    In [] !python  log.py  log_192-20171220-0826_prune.txt  step
    In [] !python  log.py  log_192-20171220-0826_acc.txt    acc
    In [] !python  log.py  log_192-20171220-0826_prune.txt  conv2rank
    In [] !python  log.py  log_192-20171220-0826_prune.txt  conv2hrank
    
    # Ipython alias
    In [] step       log_192-20171220-0826_prune.txt  
    In [] acc        log_192-20171220-0826_prune.txt
    In [] conv2rank  log_192-20171220-0826_prune.txt
'''
import os
import sys
assert (len(sys.argv) == 3)

if sys.argv[2] in ('step', 'Step'):
    if os.path.isdir(sys.argv[1]):
        prune_txts = [os.path.join(sys.argv[1], i) for i in os.listdir(sys.argv[1]) if i.endswith("prune.txt")]
        if len(prune_txts) != 1:
            print("There are not 1 `prune.txt` in your provided directory, please check.")
            print (prune_txts)
            exit(1)
        inFile = prune_txts[0]
    else:
        inFile = sys.argv[1]
        assert(inFile.endswith("prune.txt"))
    os.system('cat %s | grep "Step" > %s' % (inFile, inFile.replace('_prune.txt', '_Step.txt')))
    
elif sys.argv[2] in ('acc'):
    os.system('cat %s | grep "accuracy = "' % sys.argv[1])

elif 'hrank' in sys.argv[2]:
    os.system('cat %s | grep "%s" > %s' % (sys.argv[1], sys.argv[2], sys.argv[1].replace('_prune.txt', '_' + sys.argv[2] + '.txt')))
    
elif 'rank' in sys.argv[2]:
    os.system('cat %s | grep "%s" > %s' % (sys.argv[1], sys.argv[2], sys.argv[1].replace('_prune.txt', '_' + sys.argv[2] + '.txt')))
