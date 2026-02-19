# import all the libraries
import numpy as np
import pandas as pd
import glob
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")   

def get_files(audio_dir):
    print(f"{'*'*60}\nGetting all the audio files for the dataset creation\n")

    files = sorted(glob.glob(audio_dir + '**/*.wav', recursive=True))

    if len(files) == 0:
        print("No audio files found.")
        return [];
    else:
        print(f"Got {len(files)} from the folder {audio_dir}\n")
        return files

def make_dataset(files, sample_rate, trim = True):
    print(f"{'*'*60}\nProcessing the audio files and extracting MFCC\n")
    dataset = []
    for file in files:
        data, sr = librosa.load(file, sr = sample_rate)

        if trim:
            data, _ = librosa.effects.trim(data, top_db = 10)
            data = librosa.util.fix_length(data, size = 24000)
        mfcc = librosa.feature.mfcc(y = data, sr = sr, n_mfcc = 20)
        label = file.split('/')[-1].split('_')[0]
        label = int(label)
        dataset.append([mfcc, label])
    print("Processed and extracted MFCC.")
    return dataset

def export_tflite(model_tf):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_tf)
    tflite_model = converter.convert()
    
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print(f"{'*'*60}\nSaved the model as model.tflite\n{'*'*60}") 



if __name__ == "__main__":
    files = get_files('./free-spoken-digit-dataset-master/recordings')
    dataset = make_dataset(files, 48000)
    df = pd.DataFrame(np.array(dataset, dtype=object).squeeze(), columns=['mfcc','label'])
    print(f"{'*'*60}\nPrinting head of the created Dataset")
    print(df.head())


    x = np.array(df.mfcc.to_list())
    y = np.array(df.label.to_list())


    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    model_tf = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, 47, 20)),
        tf.keras.layers.Conv2D(32, kernel_size=(1, 3), padding='same', use_bias=True),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Conv2D(64, kernel_size=(1, 3), padding='same', use_bias=True),
        tf.keras.layers.ReLU(max_value=6),
    
        tf.keras.layers.GlobalAveragePooling2D(),
    
        tf.keras.layers.Dense(64, use_bias=True),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Dense(10, activation=None),
        tf.keras.layers.Softmax()
    ])
    
    model_tf.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    x_trn, x_val, y_trn, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_trn_reshaped = x_trn[:, np.newaxis, :, :].transpose(0, 1, 3, 2)
    x_val_reshaped = x_val[:, np.newaxis, :, :].transpose(0, 1, 3, 2)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_trn_reshaped, y_trn))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val_reshaped, y_val))

    train_dataset = train_dataset.shuffle(buffer_size=len(x_trn)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    print(f"{'*'*60}\nTraining the model for 50 epochs\n")

    model_tf.fit(train_dataset, validation_data=val_dataset, epochs=50)

    x_test_reshaped = x_test[:, np.newaxis, :, :].transpose(0, 1, 3, 2)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_reshaped, y_test)).batch(32)

    y_pred = model_tf.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    test_accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"{'*'*60}\nModel Test Accuracy: {test_accuracy:.4f}\n{'*'*60}\n")

    export_tflite(model_tf)


