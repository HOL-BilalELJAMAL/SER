"""
Module that build a model to recognize emotion from speech using the librosa
and sklearn libraries and the RAVDESS dataset.
"""

# Imports
import librosa
import soundfile
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Extract Feature Function
def extract_feature(file_name, mfcc, chroma, mel):
    """
    Function that extracts features from a sound file

    Args:
        file_name (str): Audio File Name
        mfcc (bool): Mel Frequency Cepstral Coefficient represents the
        short-term power spectrum of a sound
        chroma (bool): Pertains to the 12 different pitch classes
        mel (bool): Mel Spectrogram Frequency

    Returns:
        The result (np_array)
        For each feature of the three, if it exists, make a call to the
        corresponding function from librosa.feature and get the
        mean value.
        Call the function hstack() from numpy with result and the feature
        value, and store this in result. hstack() stacks arrays in
        sequence horizontally.
    """
    with soundfile.SoundFile(file_name) as sound_file:
        my_file = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc
                            (y=my_file, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(my_file))
            chroma = np.mean(librosa.feature.chroma_stft
                             (S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram
                          (my_file, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result


# Emotions in the RAVDESS dataset
emotions = {'01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
            }

# Emotions to Observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']


# Load Data Function
def load_data(path, test_size=0.2):
    """
    Function that loads the data and extract features for each sound file
    """
    x, y = [], []
    for file in glob.glob(path):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size,
                            random_state=9)


# Define the Audio Files Path
files_path = "C:\\ravdess data\\Actor_*\\*.wav"

# Split the Dataset
x_train, x_test, y_train, y_test = load_data(files_path, test_size=0.25)

# Get the Shape of the Training and Testing Datasets
print((x_train.shape[0], x_test.shape[0]))

# Get the Number of Features Extracted
print(f'Features extracted: {x_train.shape[1]}')

# Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,),
                      learning_rate='adaptive',
                      max_iter=500)

# Train the Model
model.fit(x_train, y_train)

# Predict for the Test Set
y_pred = model.predict(x_test)

# Calculate the Accuracy of our Model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the Accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
