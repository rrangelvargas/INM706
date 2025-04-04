import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from skimage.transform import resize

classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
directory = 'data/genres_original'


def normalize_peak(audio):
    return audio / np.max(np.abs(audio))

def normalize_standard(audio):
    return (audio - np.mean(audio)) / np.std(audio)

def preprocess_audio_files(directory, genres, resized_shape=(150, 150)):
    spectrograms = []
    genre_labels = []
    
    for genre_index, genre in enumerate(genres):        
        genre_directory = os.path.join(directory, genre)
        for file in os.listdir(genre_directory):
            if file.lower().endswith('.wav'):
                audio_path = os.path.join(genre_directory, file)
                audio, rate = librosa.load(audio_path, sr=None)
                audio = normalize_peak(audio)
                
                duration_of_chunk = 4
                overlap = 2
                samples_per_chunk = duration_of_chunk * rate
                samples_overlap = overlap * rate
                total_chunks = int(np.ceil((len(audio) - samples_per_chunk) / (samples_per_chunk - samples_overlap))) + 1
                
                for chunk_number in range(total_chunks):
                    start_sample = chunk_number * (samples_per_chunk - samples_overlap)
                    end_sample = start_sample + samples_per_chunk
                    audio_chunk = audio[start_sample:end_sample]
                    mel_spect = librosa.feature.melspectrogram(y=audio_chunk, sr=rate)
                    resized_mel_spect = resize(np.expand_dims(mel_spect, axis=-1), resized_shape)
                    spectrograms.append(resized_mel_spect)
                    genre_labels.append(genre_index)
    
    return np.array(spectrograms), np.array(genre_labels)

def get_train_test_split(train_size=0.8, test_size=0.2, random_state=42):
    print("Preprocessing audio files...")
    spectrograms, genre_labels = preprocess_audio_files(directory, classes)

    X_train, X_test, y_train, y_test = train_test_split(
        spectrograms,
        genre_labels,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state
    )

    print("Shape of the training set:", X_train.shape)
    print("Shape of the testing set:", X_test.shape)
    print("Shape of the training labels:", y_train.shape)
    print("Shape of the testing labels:", y_test.shape)

    return X_train, X_test, y_train, y_test

def load_dataset(X_train, X_test, y_train, y_test, batch_size=32):
    print("Loading dataset...")

    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Dataset loaded successfully.\n")

    return train_loader, test_loader
