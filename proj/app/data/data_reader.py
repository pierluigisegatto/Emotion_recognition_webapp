#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 15:04:57 2021

@author: segatto
"""
""" This file processes the emoDB dataset downloaded from
http://emodb.bilderbar.info/download/download.zip """


# utility for managing emotion classes and labels




import os
import numpy as np
import librosa
import soundfile
import random
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
def get_maps(from_='code', to_='label'):
    """ Returns the map from keys to items specified in the argument.
    code = W,L,E,A,F,T,N
    label = 0,1,2,3,4,5,6
    emotion = angry, boredom, disgust, fear, happy, sad, neutral
    """

    code_to_emotion = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happy",
        "T": "sad",
        "N": "neutral"
    }

    code_to_label = {v: i for v, i in zip(code_to_emotion.keys(),
                                          range(len(code_to_emotion.keys())))}

    label_to_code = {v: k for k, v in code_to_label.items()}

    emotion_to_code = {v: k for k, v in code_to_emotion.items()}

    label_to_emotion = {v: i for v, i in zip(
        code_to_label.values(), emotion_to_code.keys())}

    if from_ == 'code' and to_ == 'label':
        return code_to_label
    elif from_ == 'label' and to_ == 'code':
        return label_to_code
    elif from_ == 'code' and to_ == 'emotion':
        return code_to_emotion
    elif from_ == 'emotion' and to_ == 'code':
        return emotion_to_code
    elif from_ == 'label' and to_ == 'emotion':
        return label_to_emotion


def get_labels(DATA_PATH='data/wav/'):
    """ returns a list with filenames and the corresponding lables """
    labels = []
    filenames = []
    code_to_label = get_maps(from_='code', to_='label')

    datalist = os.listdir(DATA_PATH)
    for file in datalist:
        labels.append(code_to_label[file[5]])
        filenames.append(DATA_PATH + file)

    return filenames, labels


def check_input(DATA_PATH='data/wav/'):
    """ Checks that the input audio are the correct ones. Returns true if so."""
    audiofiles = os.listdir(DATA_PATH)
    for file in audiofiles:
        assert len(file) == len('16a02Lb.wav') or len(
            file) == len('16a02Lbaug0.wav'), 'expecting a filename of length 6 or 10, check the input'
        assert file[5] in get_maps(
            from_='code', to_='label').keys(), 'wrong input files'
    return True


def augment_sample(file, n_modifications=3, dump=True,
                   save_path='data/data_augment/'):
    """Produce n_modified versions of the input audio file. The sample is shifted,
    stretched and white noise is added. Audio files can also be saved.

    As in the reference paper:
        'Our data augmentation techniques include alterations like moving the
        beginning of the sound file by some small amount,speeding up and
        slowing down the file by 1.23% and 0.81% of its normal speed,
        respectively, and adding random noise to the 25% of its length.
    """
    sample, sr = librosa.load(file, sr=None)
    base = os.path.basename(file)[:-4]
    code_to_label = get_maps(from_='code', to_='label')
    label = code_to_label[base[5]]

    new_samples = []

    for i in range(n_modifications):
        # white noise addition
        sample_noise = sample + 0.05*np.random.normal(0, 1, len(sample))

        # time shifting: delay the beginning of the sound
        delay = random.randint(0, 6200)
        sample_shift = np.pad(sample_noise, (delay, 0),
                              mode='constant', constant_values=0)

        # time stretching
        choice = random.random()
        if choice > 0.5:
            factor = 1.23  # fast
        else:
            factor = 0.81  # slow
        sample_stetched = librosa.effects.time_stretch(sample_shift, factor)

        new_samples.append(sample_stetched)

        if dump:
            soundfile.write(
                save_path+base+'aug'+str(i)+'.wav', sample_stetched, sr)

    return new_samples, label


def get_features(file):
    """Takes a filename and returns a concatenated array of the features extracted
    using the librosa library"""
    # load audio file
    sample, sr = librosa.load(file, sr=None)
    # calculate the Mel-frequency cepstral coefficients (MFCCs)
    mfcc = np.mean(librosa.feature.mfcc(y=sample, sr=sr, n_mfcc=50).T, axis=0)

    # mel-frequency cepstrum (MFC): short-term power spectrum of a sound
    melspectrogram = np.mean(librosa.feature.melspectrogram(
        y=sample, sr=sr, n_mels=50).T, axis=0)

    # Spectral contrast
    stft = np.abs(librosa.stft(sample))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sr, n_fft=100).T, axis=0)

    # Chromagram
    chroma = librosa.feature.chroma_stft(
        S=stft, sr=sr, n_chroma=80)

    # Tonnetz (tonal centroid features)
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(sample), sr=sr, chroma=chroma).T, axis=0)

    # Averaged chromatogram
    chroma = np.mean(chroma.T, axis=0)

    # concatenate all features to get a 193 long vector
    features = np.concatenate(
        [mfcc, chroma, melspectrogram, spectral_contrast, tonnetz], axis=0)
    features = np.expand_dims(features, axis=0).astype(np.float32)
    return features


def one_hot_encode_emoDB_labels(y):
    """ one hot encode the labels"""
    lb = LabelEncoder()
    y = np_utils.to_categorical(lb.fit_transform(y.ravel()))
    return y


def dump_to_csv(X, fname):
    """ Save array to csv"""
    np.savetxt(fname, X, delimiter=",")


def create_train_test_sets(X, y, test_size=0.2, random_state=1, shuffle=True):
    """split into training and test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def check_app_paths():
    """ Makes sure that data/cached and data/data_augment folders are present"""
    # create the 'proj/app/data/cached_features/' folder
    if not os.path.isdir('data/cached/'):
        os.mkdir('data/cached/')

    # create the 'proj/app/data/data_augment/' folder
    if not os.path.isdir('data/data_augment/'):
        os.mkdir('data/data_augment/')


def get_dataset(augment=True, n_modifications=2):
    # check the paths
    check_app_paths()
    print('Dealing with the emoDB original dataset ...')

    # check if cached features exist
    if not os.path.exists('data/cached/emodB_features.csv') or not os.path.exists('data/cached/emodB_labels.csv'):
        # get list of emoDB filenames and extract their label
        # check input
        if check_input(DATA_PATH='data/wav/'):
            print('Reading emoDB original audio files ...')

            # get names and labels
            filenames, labels = get_labels(DATA_PATH='data/wav/')

            # how many audio files are present
            n_emoDB = len(filenames)

            # reshape labels to be a 2D array
            labels = np.array(labels)
            labels.shape = (n_emoDB, 1)

            # initialize the feature matrix and label vector
            X_emoDB = np.zeros((n_emoDB, 193))
            y_emoDB = labels  # assign the raw labels

            # get all features for each input file
            print('Extracting features from emoDB original audio files ...')
            for idx, file in enumerate(filenames):
                # extract features
                X_emoDB[idx, :] = get_features(file)

            # dump to csv
            print('Dumping features and labels to cache ...')
            dump_to_csv(X_emoDB, 'data/cached/emodB_features.csv')
            dump_to_csv(y_emoDB, 'data/cached/emodB_labels.csv')

    else:
        print('Using cached preprocessed results: ...')
        # load cached features and labels
        X_emoDB = np.genfromtxt(
            'data/cached/emodB_features.csv', delimiter=',')
        y_emoDB = np.genfromtxt(
            'data/cached/emodB_labels.csv', delimiter=',')
        y_emoDB.shape = (len(y_emoDB), 1)

    if augment:
        print("Dealing with the augmented Dataset ...")

        if not os.path.exists('data/cached/augmented_features.csv') or not os.path.exists('data/cached/augmented_labels.csv'):
            if not os.listdir('data/data_augment/'):
                print("Directory is empty: Populating folder with new samples")

                # create and save the augmented audio files
                for idx, file in enumerate(filenames):
                    _, _ = augment_sample(file, n_modifications, dump=True,
                                          save_path='data/data_augment/')
            else:
                print("Directory is not empty: Using already present augmented files")
                if check_input(DATA_PATH='data/data_augment/'):
                    print('Detected files are Valid')

            # read the augmented set names and get the appropriate labels
            augmented_filenames, augmented_labels = get_labels(
                DATA_PATH='data/data_augment/')
            n_augmented = len(augmented_filenames)
            # reshape labels to be a 2D array
            augmented_labels = np.array(augmented_labels)
            augmented_labels.shape = (n_augmented, 1)

            # initialize the feature matrix and label vector
            X_augmented = np.zeros((n_augmented, 193))
            y_augmented = augmented_labels  # assign the raw labels

            # get all features of the augmented set
            print('Extracting features from the augmented audio files ...')
            for idx, file in enumerate(augmented_filenames):
                # extract features
                X_augmented[idx, :] = get_features(file)

            # dump to csv
            print('Dumping augmented features and labels to cache ...')
            dump_to_csv(
                X_augmented, 'data/cached/augmented_features.csv')
            dump_to_csv(
                y_augmented, 'data/cached/augmented_labels.csv')
        else:
            print('Using cached preprocessed results: ...')
            # load cached features and labels
            X_augmented = np.genfromtxt(
                'data/cached/augmented_features.csv', delimiter=',')
            y_augmented = np.genfromtxt(
                'data/cached/augmented_labels.csv', delimiter=',')
            y_augmented.shape = (len(y_augmented), 1)

        # concatenate the emoDB and augmented sets
        X_emoDB = np.vstack((X_emoDB, X_augmented))
        y_emoDB = np.vstack((y_emoDB, y_augmented))

    # one hot encode the labels
    y_emoDB = one_hot_encode_emoDB_labels(y_emoDB)

    return X_emoDB, y_emoDB


if __name__ == '__main__':
    # the following is the pipeline to get all files red, augmented, processed,
    # and separated into train and test set
    import time
    os.chdir('../')

    t = time.time()
    random.seed(time.time())
    X, y = get_dataset(augment=True, n_modifications=3)
    X_train, X_test, y_train, y_test = create_train_test_sets(
        X, y, test_size=0.2, random_state=1, shuffle=True)
    elapsed1 = time.time() - t
