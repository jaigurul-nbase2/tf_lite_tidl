# import all the libraries
import numpy as np
import pandas as pd
import glob
import librosa

def get_files(audio_dir):
    print("*"*60)
    print("Getting all the audio files for the dataset creation")
    print("\n")

    files = sorted(glob.glob(audio_dir + '**/*.wav', recursive=True))

    if len(files) == 0:
        print("No audio files found.")
        return [];
    else:
        print(f"Got {len(files)} from the folder {audio_dir}")
        print("*"*60)
        return files

def make_dataset(files, sample_rate, trim = True):
    dataset = []
    for file in files:
        data, sr = librosa.load(file, sr = sample_rate)

        if trim:
            data, _ = librosa.effects.trim(data, top_db = 10)
            data = librosa.util.fix_length(data, size = 24000)
        mfcc = librosa.feature.mfcc(y = data, sr = sr, n_mfcc = 20)
        label = file.split('/')[-1].split('_')[0]
        dataset.append([mfcc, label])
    return dataset


if __name__ == "__main__":
    files = get_files('./free-spoken-digit-dataset-master/recordings')
    dataset = make_dataset(files, 48000)
    df_ds = pd.DataFrame(np.array(dataset, dtype=object).squeeze(), columns=['mfcc','label'])
    print(df_ds.head())
