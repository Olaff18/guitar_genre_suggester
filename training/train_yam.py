import os
import glob
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # <--- CHANGED BACK (See note below)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- CONFIG ---
SR_YAMNET = 16000 
# ADDED 'noise' to the list so the model learns what silence sounds like
GENRES = ['bossa', 'punk', 'doom', 'noise'] 
OUT = "training_data"

# --- LOAD YAMNET ---
print("Loading YAMNet from TFHub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def get_yamnet_embeddings(audio, sr):
    # 1. Sanitize: Ensure Float32 & Mono
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1) 
    audio = audio.flatten()

    # 2. Resample
    if sr != SR_YAMNET:
        if len(audio) == 0: return np.array([])
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_YAMNET)
    
    # 3. Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # 4. Inference
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return embeddings.numpy()

def train():
    print("Extracting YAMNet Embeddings...")
    X = []
    y = []

    for g in GENRES:
        # We look for ANY file starting with the genre name
        files = glob.glob(f"{OUT}/{g}_long.npy")
        print(f"Processing {g} ({len(files)} files found)...")
        
        if len(files) == 0:
            print(f"WARNING: No files found for {g}! Check filenames in 'training_data' folder.")
            continue

        for f in files:
            try:
                audio = np.load(f)
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")
                continue
            
            # Extract features
            embeddings = get_yamnet_embeddings(audio, sr=44100)
            
            for emb in embeddings:
                X.append(emb)
                y.append(g)

    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("ERROR: No data extracted.")
        return

    print(f"\nFeature Matrix Shape: {X.shape}") 

    # --- SPLIT ---
    # We switched back to train_test_split because you have 1 long file per genre.
    # GroupShuffleSplit would fail here (it can't split 1 file into two).
    # Since your recording is 3 mins long and varied, random splitting is safe here.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")

    # --- TRAIN ---
    clf = RandomForestClassifier(n_estimators=200, max_depth=10)
    clf.fit(X_train, y_train)

    # --- EVAL ---
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    # --- SAVE ---
    joblib.dump(clf, "genre_classifier_yamnet.pkl")
    print("Saved 'genre_classifier_yamnet.pkl'")

if __name__ == "__main__":
    train()