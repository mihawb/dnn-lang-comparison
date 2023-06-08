#!/bin/sh

url_base=http://yann.lecun.com/exdb/mnist

wget ${url_base}/train-images-idx3-ubyte.gz
wget ${url_base}/train-labels-idx1-ubyte.gz
wget ${url_base}/t10k-images-idx3-ubyte.gz
wget ${url_base}/t10k-labels-idx1-ubyte.gz

gunzip *.gz