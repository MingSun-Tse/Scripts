#coding=utf-8
from util import *

assert(len(sys.argv) == 2)
mean_file = os.path.abspath(sys.argv[1])

# 使输出的参数完全显示
# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
np.set_printoptions(threshold='nan')

# 保存参数的文件
means_txt = os.path.join(os.path.dirname(mean_file), "mean.txt")
mean_outfile = open(means_txt, 'w')

# 将均值文件读入blob中
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(mean_file, 'rb').read())

# 将均值blob转为numpy.array
mean = caffe.io.blobproto_to_array(mean_blob)
mean.shape = (-1, 1)
print("the numbers in mean: %d" % mean.shape[0])
for m in mean:
    mean_outfile.write('%f, ' % m)
mean_outfile.close
print("saved mean txt to `%s`" % means_txt)