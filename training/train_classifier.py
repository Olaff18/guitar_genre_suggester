import os
import glob
import time
import numpy as np
import sounddevice as sd # do nagrywania z mikro
import librosa # do ekstrakcji cech audio
from sklearn.ensemble import RandomForestClassifier
import joblib # do zapisywania modelu

SR = 44100 # sample rate 44.1 kHz klasyczne CD quality 
DURATION = 5.0 # 5 sekund na klip
GENRES = ['metal', 'shoegaze', 'punk', 'doom']
OUT = "training_data" # folder do zapisywania danych treningowych
os.makedirs(OUT, exist_ok=True) # jakby nie istnial folder ale istnieje

def record_clip(filename, seconds=DURATION):
    print(f"Rcording {filename} for {seconds} seconds (clean, no effects)")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait() # blokuje do konca nagrania   
    np.save(filename, audio) # zapisuje nagranie jako plik .npy
    print(f"Saved recording to {filename}")



# bierze .npy (raw waveform) i zamienia w zbiory cyferek reprezentujace dzwieki
def extract_features_np(audio_np, sr=SR):
    y = audio_np.flatten() # .npy mze byc 2D wiec spłaszczamy do 1D
    #upewniamy sie ze nonempty
    if y.size == 0:
        return None
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) # mierzy "centrum masy", wysoki centrioid = jasny dzwiek (moze metal picking), niski = ciemny (doom na pewno, shoegaze tez)
    rms = np.mean(librosa.feature.rms(y=y)) # root mean square energy, mierzy glosnosc/moc dzwieku, jak wysoki to np heavy strumming
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y)) # zero crossing rate, mierzy ilosc zmian znaku w sygnale, wysoki zcr = szumowate dzwieki (punk moze miec wiecej zcr) a niski zcr = czystsze dzwieki
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)) # mierzy rozklad czestotliwosci, szerokie pasmo = bogate harmonicznie dzwieki (metal) a waskie pasmo = proste dzwieki (doom)
    return [centroid, rms, zcr, bandwidth]

def collect_samples():
    print("Recordujemy 3 clipy per gatunek: ")
    for g in GENRES:
        for i in range(3):
            fname = f"{OUT}/{g}_{i}.npy"
            input(f"Nagrywamy {g} clip #{i+1}. Klik enter i gramy...")
            record_clip(fname)
            time.sleep(0.5)

def train():
    X, y = [], [] # X to cechy, y to etykiety gatunkow
    for g in GENRES:
        for f in glob.glob(f"{OUT}/{g}_*.npy"): # szukamy wszystkich plikow .npy dla danego gatunku *glob to po prostu do wygodnego szukania plikow
            audio = np.load(f)
            feat = extract_features_np(audio, SR) # wyciagamy cechy, cyferki
            if feat is None:
                continue
            X.append(feat)
            y.append(g)
    X = np.array(X)
    clf = RandomForestClassifier(n_estimators=100, random_state=0) # 100 drzewek, totalnie siądzie nawet z szumem mysle
    clf.fit(X, y)
    joblib.dump(clf, "genre_classifier.pkl") # zapisujemy model
    print("zapisujemy model do pliku genre_classifier.pkl")

if __name__ == "__main__":
    collect_samples()
    train()