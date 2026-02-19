# Model Development

In this folder model development is taken care of the dataset preprocessing, model training and export to the tflite

## Steps to develop the audio classification model

- Here we have used the digit classification where the input is the wav file and the output is the digit classification [0-9]
- For this step both Windows and Linux can be used to get the model.tflite

### 1. Setup the dataset which has wav files in it

In windows extract the archive.zip file to get the folder free-spoken-digit-dataset-master

```bash
# In Linux use
unzip archive.zip
```

### 2. Setup the python virtual environment for development of the model

```bash
python -m venv .venv

# Activate the virtual environment

# Windows
./.venv/Scripts/activate

# Linux
source .venv/bin/activate

# Install all the requirements
pip install -r requirements.txt
```

### 3. Run the script to load the dataset, preprocess, train and save it as tflite

```bash
python main.py
```

### 4. The model will be trained and saved as model/model.tflite


