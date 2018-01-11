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
    assert('_prune.txt' in sys.argv[1])
    os.system('cat %s | grep "Step" > %s' % (sys.argv[1], sys.argv[1].replace('_prune.txt', '_Step.txt')))
    
elif sys.argv[2] in ('acc'):
    os.system('cat %s | grep "accuracy = "' % sys.argv[1])

elif 'hrank' in sys.argv[2]:
    os.system('cat %s | grep "%s" > %s' % (sys.argv[1], sys.argv[2], sys.argv[1].replace('_prune.txt', '_' + sys.argv[2] + '.txt')))
    
elif 'rank' in sys.argv[2]:
    os.system('cat %s | grep "%s" > %s' % (sys.argv[1], sys.argv[2], sys.argv[1].replace('_prune.txt', '_' + sys.argv[2] + '.txt')))

