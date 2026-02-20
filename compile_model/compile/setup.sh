#!/bin/bash
set -e


URL="https://software-dl.ti.com/jacinto7/esd/tidl-tools/11_01_06_00/TIDL_TOOLS/AM62A/tidl_tools.tar.gz"
FILE="tidl_tools.tar.gz"

wget $URL -O $FILE

tar -xzf $FILE

rm $FILE
