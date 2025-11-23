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
from sklearn.preprocessing import StandardScaler # <--- NEW: Important for accuracy
from sklearn.model_selection import GroupShuffleSplit # <--- NEW: for group-based splitting


SR = 44100 # sample rate 44.1 kHz klasyczne CD quality 
DURATION = 5.0 # 5 sekund na klip
GENRES = ['bossa', 'punk', 'doom', 'noise']
OUT = "training_data" # folder do zapisywania danych treningowych
os.makedirs(OUT, exist_ok=True) # jakby nie istnial folder ale istnieje

def record_clip(filename, seconds=DURATION):
    print(f"Rcording {filename} for {seconds} seconds (clean, no effects)")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait() # blokuje do konca nagrania   
    np.save(filename, audio) # zapisuje nagranie jako plik .npy 
    print(f"Saved recording to {filename}")

# --- 2. NEW: SLICING FUNCTION ---
# this turns 1 file into 10+ training examples
def slice_audio(audio, sr, chunk_len=3.0, overlap=2.5):
    # chunk_len = 3.0 : enough time to hear the "Silence between notes"
    # overlap = 2.5   : we step forward only 0.5s at a time to get MORE data
    n_samples = int(chunk_len * sr)   
    step = int((chunk_len - overlap) * sr)
    
    slices = []
    for i in range(0, len(audio) - n_samples, step):
        chunk = audio[i : i + n_samples]
        if len(chunk) == 0: continue

        slices.append(chunk)
    return slices

# bierze .npy (raw waveform) i zamienia w zbiory cyferek reprezentujace dzwieki
def extract_features_np(audio_np, sr=SR):
    y = audio_np.flatten().astype(np.float32)

    # we normalize here so quiet Bossa isn't ignored
    # we use a small noise gate (0.005) to avoid boosting silence
    if np.max(np.abs(y)) < 0.005:
        return None
        
    y = librosa.util.normalize(y) 


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
        return [np.mean(x), np.std(x)] # removed median to save speed, mean/std is usually enough

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
    print("Processing data...")
    X, y = [], []
    
    for g in GENRES:
        # Matches bossa_long.npy, noise_long.npy (or noise_0.npy)
        # We use * to be flexible with names
        files = glob.glob(f"{OUT}/{g}_long.npy")
        print(f"Processing {g}: {len(files)} files found.")
        
        if len(files) == 0:
            print(f"WARNING: No files found for '{g}'! Check folder/names.")

        for f in files:
            full_audio = np.load(f)
            
            # Use 3.0s chunks with high overlap
            slices = slice_audio(full_audio, SR, chunk_len=3.0, overlap=2.5)
            
            for s in slices:
                feat = extract_features_np(s, SR)
                if feat is not None:
                    X.append(feat)
                    y.append(g)

    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTotal Training Samples: {len(X)}")

    # --- 3. SPLITTING (Switched back to standard split) ---
    # Since we have 1 long varied file per genre, we split the SLICES randomly.
    # GroupShuffleSplit is not possible with only 1 file per genre.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Increased depth slightly to handle more complex variations in long files
    clf = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== ACCURACY ===")
    print(f"{acc*100:.2f}%")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, "genre_classifier.pkl")
    joblib.dump(scaler, "genre_scaler.pkl")
    print("\nModel and Scaler saved!")


if __name__ == "__main__":
    # collect_samples()
    train()