#!/usr/bin/env sh

# Set bin
CAFFE_ROOT="/home2/wanghuan/Scripts/Python/PCA_analysis/Caffe_APP"

# Set project
PROJECT="/home2/wanghuan/Scripts/Python/PCA_analysis/MyProject0"
WEIGHTS="/home2/wanghuan/Caffe/caffe_models/convnet/baseline_iter_140000.caffemodel"
TIME="$SERVER-$(date +%Y%m%d-%H%M)"

$CAFFE_ROOT/build/tools/caffe train \
--gpu $1 \
--weights $WEIGHTS \
--solver  $PROJECT/solver.prototxt \
2>>       $PROJECT/weights/log_$TIME\_acc.txt \
1>>       $PROJECT/weights/log_$TIME\_prune.txt
