import os
import tflite_runtime.interpreter as tflite
import numpy as np
import librosa
import yaml


model_path = "model/model.tflite"
artifacts_folder = "model_artifacts/classification/artifacts"
tidl_tools_path = "tidl_tools"

calib_audios = [
    "sample_audio/0.wav",
    "sample_audio/1.wav",
    "sample_audio/5.wav",
    ]

def preprocess_audio(path):
    data, sr = librosa.load(path, sr= 48000)
    data, area = librosa.effects.trim(data, top_db=10)
    data = librosa.util.fix_length(data, size=24000)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
    mfcc_T = mfcc.T
    input_data = mfcc_T[np.newaxis,np.newaxis, :, :]
    return input_data

def gen_param_yaml():
    param_dict = {
        "task_type": "classification",
        "preprocess": {
            "data_layout": "NCHW",
            "mean": [123.675, 116.28, 103.53],
            "scale": [0.0171, 0.0175, 0.0174],
        },
        "session": {
            "session_name": "tflite",
            "model_path": model_path,
            "artifacts_folder": artifacts_folder,
            "input_data_layout": "NCHW",
            "target_device": "AM62A"
        }
    }
    with open("model_artifacts/classification/params.yaml", "w") as file:
        yaml.safe_dump(param_dict, file, sort_keys=False)

os.makedirs(artifacts_folder, exist_ok=True)
for root, dirs, files in os.walk(
    artifacts_folder, topdown=False
):
    [os.remove(os.path.join(root, f)) for f in files]
    [os.rmdir(os.path.join(root, d)) for d in dirs]


interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tflite.load_delegate(
                    os.path.join(tidl_tools_path, "tidl_model_import_tflite.so"),
                    {
                        "tidl_tools_path": tidl_tools_path,
                        "artifacts_folder": artifacts_folder,
                        "tensor_bits": 16,
                        "accuracy_level": 1,
                        "debug_level": 3,
                        "advanced_options:calibration_frames" : len(calib_audios),
                        "advanced_options:calibration_iterations": 3,
                        "advanced_options:add_data_convert_ops": 1,
                    },
                )
            ],
        )

interpreter.allocate_tensors()

# input_data = preprocess_audio("sample_audio/0.wav")



output_details = interpreter.get_output_details()
input_details = interpreter.get_input_details()


for audio in calib_audios:
    input_data = preprocess_audio(audio)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()


output = interpreter.get_tensor(output_details[0]['index'])
print(output)
gen_param_yaml()
