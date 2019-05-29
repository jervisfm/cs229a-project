#!/usr/bin/env bash

mkdir -p data/matlab
mkdir -p results
mkdir -p imgs/cm_imgs
wget "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip" -O data/matlab.zip
unzip data/matlab.zip -d data/
