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


audio1 = wave.open("barwav/barok1.wav")
audio2 = wave.open("barwav/barok2.wav")
audio3 = wave.open("barwav/barok3.wav")
audio4 = wave.open("barwav/barok4.wav")
audio5 = wave.open("barwav/barok5.wav")


barok1 = "barokdata/barok_1.npy"
barok2 = "barokdata/barok_2.npy"
barok3 = "barokdata/barok_3.npy"
barok4 = "barokdata/barok_4.npy"
barok5 = "barokdata/barok_5.npy"




a1 = wave.open("doom_1.wav")
a2 = wave.open("doom_2.wav")
a3 = wave.open("doom_3.wav")
a4 = wave.open("doom_4.wav")
a5 = wave.open("doom_5.wav")
a6 = wave.open("doom_6.wav")
a7 = wave.open("doom_7.wav")
a8 = wave.open("doom_8.wav")
a9 = wave.open("doom_9.wav")
a10 = wave.open("doom_10.wav")

b1 = "training_data/doom_1.npy"
b2 = "training_data/doom_2.npy"
b3 = "training_data/doom_3.npy"
b4 = "training_data/doom_4.npy"
b5 = "training_data/doom_5.npy"
b6 = "training_data/doom_6.npy"
b7 = "training_data/doom_7.npy"
b8 = "training_data/doom_8.npy"
b9 = "training_data/doom_9.npy"
b10 = "training_data/doom_10.npy"


c1 = wave.open("punk_1.wav")
c2 = wave.open("punk_2.wav")
c3 = wave.open("punk_3.wav")
c4 = wave.open("punk_4.wav")
c5 = wave.open("punk_5.wav")
c6 = wave.open("punk_6.wav")
c7 = wave.open("punk_7.wav")
c8 = wave.open("punk_8.wav")
c9 = wave.open("punk_9.wav")
c10 = wave.open("punk_10.wav")

d1 = "training_data/punk_1.npy"
d2 = "training_data/punk_2.npy"
d3 = "training_data/punk_3.npy"
d4 = "training_data/punk_4.npy"
d5 = "training_data/punk_5.npy"
d6 = "training_data/punk_6.npy"
d7 = "training_data/punk_7.npy"
d8 = "training_data/punk_8.npy"
d9 = "training_data/punk_9.npy"
d10 = "training_data/punk_10.npy"




np.save(barok1, np.frombuffer(audio1.readframes(audio1.getnframes()), dtype=np.int16))
np.save(barok2, np.frombuffer(audio2.readframes(audio2.getnframes()), dtype=np.int16))
np.save(barok3, np.frombuffer(audio3.readframes(audio3.getnframes()), dtype=np.int16))
np.save(barok4, np.frombuffer(audio4.readframes(audio4.getnframes()), dtype=np.int16))
np.save(barok5, np.frombuffer(audio5.readframes(audio5.getnframes()), dtype=np.int16))


# np.save(b1, np.frombuffer(a1.readframes(a1.getnframes()), dtype=np.int16))
# np.save(b2, np.frombuffer(a2.readframes(a2.getnframes()), dtype=np.int16))
# np.save(b3, np.frombuffer(a3.readframes(a3.getnframes()), dtype=np.int16))
# np.save(b4, np.frombuffer(a4.readframes(a4.getnframes()), dtype=np.int16))
# np.save(b5, np.frombuffer(a5.readframes(a5.getnframes()), dtype=np.int16))
# np.save(b6, np.frombuffer(a6.readframes(a6.getnframes()), dtype=np.int16))
# np.save(b7, np.frombuffer(a7.readframes(a7.getnframes()), dtype=np.int16))
# np.save(b8, np.frombuffer(a8.readframes(a8.getnframes()), dtype=np.int16))
# np.save(b9, np.frombuffer(a9.readframes(a9.getnframes()), dtype=np.int16))
# np.save(b10, np.frombuffer(a10.readframes(a10.getnframes()), dtype=np.int16))

# np.save(d1, np.frombuffer(c1.readframes(c1.getnframes()), dtype=np.int16))
# np.save(d2, np.frombuffer(c2.readframes(c2.getnframes()), dtype=np.int16))
# np.save(d3, np.frombuffer(c3.readframes(c3.getnframes()), dtype=np.int16))
# np.save(d4, np.frombuffer(c4.readframes(c4.getnframes()), dtype=np.int16))
# np.save(d5, np.frombuffer(c5.readframes(c5.getnframes()), dtype=np.int16))
# np.save(d6, np.frombuffer(c6.readframes(c6.getnframes()), dtype=np.int16))
# np.save(d7, np.frombuffer(c7.readframes(c7.getnframes()), dtype=np.int16))
# np.save(d8, np.frombuffer(c8.readframes(c8.getnframes()), dtype=np.int16))
# np.save(d9, np.frombuffer(c9.readframes(c9.getnframes()), dtype=np.int16))
# np.save(d10, np.frombuffer(c10.readframes(c10.getnframes()), dtype=np.int16))
# np.save(filename, audio) # zapisuje nagranie jako plik .npy
