# download images and annotations for iChallange-AMD from
# https://ai.baidu.com/broad/download

#!/bin/bash

unzip AMD-Training400.zip
unzip DF-Annotation-Training400.zip

mkdir Training1200
mkdir Training1200/AMD
mkdir Training1200/Non-AMD

python transform.py

rm -r Training400/