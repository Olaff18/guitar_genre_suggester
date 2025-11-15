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


SR = 44100 # sample rate 44.1 kHz klasyczne CD quality 
DURATION = 5.0 # 5 sekund na klip
GENRES = ['bossa_nova', 'punk', 'doom', 'barok']
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
    y = audio_np.flatten().astype(np.float32)

    if y.size == 0:
        return None

    #  min length to avoid librosa crashes
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    # BASIC FEATURES (3 stats each)
   
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # rozne statistics oprocz mean
    def stats(x):
        return [np.mean(x), np.std(x), np.median(x)]

    feat_centroid = stats(centroid)
    feat_rms = stats(rms)
    feat_zcr = stats(zcr)
    feat_bandwidth = stats(bandwidth)

    # MFCCs (13 coefficients)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    # spectral contrast (7 bands)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_means = np.mean(contrast, axis=1)


    # rolloff (shape of spectrum)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feat_rolloff = stats(rolloff)

    # FINAL FEATURE VECTOR
    
    final_features = np.concatenate([
        feat_centroid,
        feat_rms,
        feat_zcr,
        feat_bandwidth,
        mfcc_means,
        mfcc_stds,
        contrast_means,
        feat_rolloff
    ])

    return final_features


def collect_samples():
    print("Recordujemy 5 clipy per gatunek: ")
    for g in GENRES:
        for i in range(5):
            fname = f"{OUT}/{g}_{i}.npy"
            input(f"Nagrywamy {g} clip #{i+1}. Klik enter i gramy...")
            record_clip(fname)
            time.sleep(0.5)

def train():
    X, y = [], []  # X to cechy, y to etykiety gatunkow
    for g in GENRES:
        for f in glob.glob(f"{OUT}/{g}_*.npy"):
            audio = np.load(f)
            feat = extract_features_np(audio, SR)
            if feat is None:
                continue
            X.append(feat)
            y.append(g)

    X = np.array(X)
    y = np.array(y)

    # ------------------------
    # TRAIN/TEST SPLIT
    # ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # ------------------------
    # EVALUATION
    # ------------------------
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== ACCURACY ===")
    print(f"{acc*100:.2f}%")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(clf, "genre_classifier.pkl")
    print("zapisujemy model do pliku genre_classifier.pkl")


if __name__ == "__main__":
    # collect_samples()
    train()