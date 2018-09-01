#!/usr/bin/env sh

# Set bin
CAFFE_ROOT=""

# Set project
PROJECT=""
WEIGHTS=""
TIME="$SERVER-$(date +%Y%m%d-%H%M)"

$CAFFE_ROOT/build/tools/caffe train \
--gpu $1 \
--weights $WEIGHTS \
--solver  $PROJECT/solver.prototxt \
2>>       $PROJECT/weights/log_$TIME\_acc.txt \
1>>       $PROJECT/weights/log_$TIME\_prune.txt
