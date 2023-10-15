#!/bin/bash

find $(dirname $0) ! -name $(basename $0) -type f -exec rm -f {} +

wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P $(dirname $0)

gunzip cifar-10-binary.tar.gz
tar -xvf cifar-10-binary.tar
rm cifar-10-binary.tar