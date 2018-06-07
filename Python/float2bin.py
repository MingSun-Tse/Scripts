import sys
import os
assert(len(sys.argv) == 2)
"""Usage:
    python  this_file.py  0.2435
"""

integer  = "0."
fraction = float(sys.argv[1])

while fraction:
    fraction *= 2
    integer  += str(int(fraction // 1))
    fraction -= fraction // 1
    print(integer, fraction)
