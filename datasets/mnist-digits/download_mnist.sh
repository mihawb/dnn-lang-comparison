#!/bin/bash

find $(dirname $0) ! -name $(basename $0) -type f -exec rm -f {} +

url_base=http://yann.lecun.com/exdb/mnist

wget ${url_base}/train-images-idx3-ubyte.gz -P $(dirname $0)
wget ${url_base}/train-labels-idx1-ubyte.gz -P $(dirname $0)
wget ${url_base}/t10k-images-idx3-ubyte.gz -P $(dirname $0)
wget ${url_base}/t10k-labels-idx1-ubyte.gz -P $(dirname $0)

gunzip -r "$(dirname $0)"