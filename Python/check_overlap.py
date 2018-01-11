import check_zero.check_zeros as check_zero
import sys

def get_coincidence(array1, array2):
    cnt = 0
    for i in range(len(array1)):
        if array1[i] in array2 and array1[i] in array3:
            cnt += 1
    return (cnt, 
            float(cnt) / len(array1), 
            float(cnt) / len(array2), 
            float(cnt) / len(array3))

    
def check_overlap(model, weights1, weights2, layer):
    ''' 
        As a method to select cols to prune, how robust is SSL?
        Does it always choose roughly the same cols in different runs?
    '''
    array1 = check_zero(model, weights1, layer)
    array2 = check_zero(model, weights2, layer)
    co_num, co_ratio1, co_ratio2, co_ratio3 = get_coincidence(array1, array2)
    
    print ("ave of the same cols: %d (%.3f, %.3f)" % (co_num, co_ratio1, co_ratio2))

    
### main
check_overlap(*sys.argv[1:])