# Compilation of the TfLite model to Artifacts

## 1. Setup python3.10
```bash
# Check the current version
python3 --version
```

If the version is not 3.10 then install python3.10
```bash
# Install python3.10
sudo update
sudo apt install python3.10
```
Create a virtual environment for the installation of the requirements
```bash
# Create venv
python3.10 -m venv venv

# Activate the venv
source venv.bin.activate
```

## 2. Run the setup script to install required packages and tidl tools
```bash
# Give permission to execute
chmod +x ./setup.sh
./setup.sh
```

## 3. Run the compilation script with the exported model in the directory model/model.tflite
```bash
python3 main.py
```








### MODEL CONFIGURATION NEEDS

 - task_type
 - source
 - preprocess
 - session
 - runtime_options
 - postprocess
 - metric
 - extra_info
 - model_info

### SET INPUT DATA (IMAGES)

### DELEGATE OPTIONS

- Required options : *tidl_tools_path* and *artifacts_folder*

### CLEAN THE ARTIFACTS FOLDER

### SETUP THE INTERPRETER AND GET INPUT AND OUTPUT DETAILS FROM THE MODEL

### CALL THE INFERER with Interpreter, Input_data, Config.

Preprocess the input data, allocates the tensors, set the tensor and invokes the interpreter.

outputs the output tensor

### GENERATE PARAMS.YAML

# Model compilation to generate the artifacts that can be loaded to the DSP of target

Here the input is .tflite model which is compiled to the artifacts which finally should have the following 3 files in the artifacts folder

- *io.bin
- *net.bin
- AllowedNodes.txt

