import os
import tflite_runtime.interpreter as tflite
import numpy as np
import librosa
import sys

model_path = "model/model.tflite"
artifacts_folder = "model_artifacts/classification/artifacts"

# Inference options
inference_options = {
    'tidl_tools_path': os.path.abspath("./tidl_tools"),
    'artifacts_folder': os.path.abspath(artifacts_folder),
    'debug_level': 2,
    'platform': 'PC',
}

print("Loading TIDL delegate and model...")
delegate = tflite.load_delegate('./tidl_tools/libtidl_tfl_delegate.so', inference_options)

interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[delegate],
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✓ Model loaded successfully!")
print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Output shape: {output_details[0]['shape']}")

def preprocess_audio(path):
    data, sr = librosa.load(path, sr=48000)
    data, _ = librosa.effects.trim(data, top_db=10)
    data = librosa.util.fix_length(data, size=24000)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
    mfcc_T = mfcc.T
    input_data = mfcc_T[np.newaxis, np.newaxis, :, :]
    return input_data.astype(np.float32)

# Run inference
test_audio = "sample_audio/1.wav"
if os.path.exists(test_audio):
    input_data = preprocess_audio(test_audio)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    confidence = output[0][predicted_class]
    
    print(f"\n✓ Inference complete!")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  All probabilities: {output[0]}")
else:
    print(f"⚠ Test audio not found: {test_audio}")