import sys
import os

'''Usage:
    Ipython alias usage:
      [] vv 1123a  # a: acc p: prune, a and p must NOT be absent!

    This will turn into:
      python  this_file.py  1123acc

    Note:
      This command can ONLY be used in current directory which includes the acc_txt file.
'''
assert(len(sys.argv) == 2)
key_words = sys.argv[1] #TODO: generalize this to unlimited number of argv
assert('a' in key_words or 'p' in key_words)
k1 = 'a' if 'a' in key_words else 'p'
time = key_words.split(k1)[0]

inFile = [i for i in os.listdir('.') if time in i and k1 in i]
assert(len(inFile) == 1)
os.system('vim %s' % inFile[0])
