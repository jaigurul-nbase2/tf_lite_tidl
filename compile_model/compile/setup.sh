#!/bin/bash
set -e

#!/bin/bash

echo "========================================================================="
echo "Installing all required packages for python"
echo "========================================================================="

REL=11_01_06_00
BASE_URL="https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04"

pip3 install pybind11[global]
pip3 install -r ./requirements_pc.txt

pip3 install --quiet $BASE_URL/dlr-1.13.0-py3-none-any.whl
pip3 install --quiet $BASE_URL/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
pip3 install --quiet $BASE_URL/onnxruntime_tidl-1.15.0-cp310-cp310-linux_x86_64.whl
pip3 install --quiet $BASE_URL/tflite_runtime-2.12.0-cp310-cp310-linux_x86_64.whl

echo "========================================================================="
echo "All packages installed successfully!"
echo "========================================================================="

URL="https://software-dl.ti.com/jacinto7/esd/tidl-tools/11_01_06_00/TIDL_TOOLS/AM62A/tidl_tools.tar.gz"
FILE="tidl_tools.tar.gz"


wget $URL -O $FILE

tar -xzf $FILE

rm $FILE

echo "========================================================================="
echo "Successfully downloaded and extracted the TIDL TOOLS"
echo "========================================================================="
