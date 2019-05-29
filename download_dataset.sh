#!/usr/bin/env bash

mkdir -p data/matlab
wget "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip" -O data/matlab.zip
unzip data/matlab.zip -d data/