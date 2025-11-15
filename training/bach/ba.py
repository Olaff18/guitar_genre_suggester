import os
import glob
import time
import numpy as np
import sounddevice as sd # do nagrywania z mikro
import librosa # do ekstrakcji cech audio
from sklearn.ensemble import RandomForestClassifier
import joblib # do zapisywania modelu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wave

SR = 44100 # sample rate 44.1 kHz klasyczne CD quality 
DURATION = 5.0 # 5 sekund na klip
GENRES = ['bossa_nova', 'punk', 'doom']
OUT = "training_data" # folder do zapisywania danych treningowych
os.makedirs(OUT, exist_ok=True) # jakby nie istnial folder ale istnieje


audio1 = wave.open("barok1.wav")
audio2 = wave.open("barok2.wav")
audio3 = wave.open("barok3.wav")
audio4 = wave.open("barok4.wav")
audio5 = wave.open("barok5.wav")
barok1 = "training_data/barok1.npy"
barok2 = "training_data/barok2.npy"
barok3 = "training_data/barok3.npy"
barok4 = "training_data/barok4.npy"
barok5 = "training_data/barok5.npy"
np.save(barok1, np.frombuffer(audio1.readframes(audio1.getnframes()), dtype=np.int16))
np.save(barok2, np.frombuffer(audio2.readframes(audio2.getnframes()), dtype=np.int16))
np.save(barok3, np.frombuffer(audio3.readframes(audio3.getnframes()), dtype=np.int16))
np.save(barok4, np.frombuffer(audio4.readframes(audio4.getnframes()), dtype=np.int16))
np.save(barok5, np.frombuffer(audio5.readframes(audio5.getnframes()), dtype=np.int16))
# np.save(filename, audio) # zapisuje nagranie jako plik .npy
