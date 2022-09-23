import os

import librosa
import librosa.display

import numpy as np
import config

import gc
import matplotlib.pyplot as plt
from PIL import Image


def check_signalExtension(y, fs):

    minNumOfSamples = config._minSignalLength*fs

    if len(y) < minNumOfSamples:

        y = np.tile(y, int(np.ceil(minNumOfSamples/len(y))))[:minNumOfSamples]

    return y


def compute_melSpectrogram(y, fs):

    y = check_signalExtension(y, fs)

    S = librosa.feature.melspectrogram(y=y, sr=fs, hop_length=config._hop_size, n_mels=128)
    S = (S - S.min()) / (S.max() - S.min() + np.finfo('float').eps)

    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + np.finfo('float').eps)

    return S_dB


def generate_spectrograms(signalPath, outputPath):

    generatedFiles = []

    y, fs = librosa.load(signalPath, sr = config._fs, mono=True)
    
    representation = compute_melSpectrogram(y, fs)

    samplesInTempBin = int(np.ceil((config._minSignalLength*fs)/config._hop_size))

    for ID, binID in enumerate(np.arange(0, representation.shape[1], int(np.ceil(config._binOverlap*samplesInTempBin)))):

        if binID + samplesInTempBin <= representation.shape[1]:

            fig = plt.figure()

            plt.axis('off')
            plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])

            librosa.display.specshow(representation[:,binID:binID+samplesInTempBin], y_axis='mel', x_axis='time', sr=fs, hop_length=config._hop_size)

            plt.savefig('tmp.png', bbox_inches=None, pad_inches=0)

            plt.close(fig)
            gc.collect()

            spectrogramFileName = 'frame{:03d}.png'.format(ID+1)

            img = Image.open('tmp.png')
            img = img.resize((config._spectrogramSize, config._spectrogramSize), Image.ANTIALIAS)
            img.save(os.path.join(outputPath, spectrogramFileName))

            os.system('rm -f tmp.png')

            generatedFiles.append(os.path.join(outputPath, spectrogramFileName))

    return generatedFiles

