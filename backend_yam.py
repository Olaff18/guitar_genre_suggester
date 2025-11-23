import os
import tempfile
import time
import logging
from threading import Thread
from flask import Flask, jsonify, request
import numpy as np
import queue
import sounddevice as sd
import joblib
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# Check for Pedalboard
try:
    from pedalboard import Pedalboard, Distortion, Gain, Reverb, LowpassFilter, HighpassFilter, Convolution
except ImportError:
    print("Warning: Pedalboard not found. Effects disabled.")
    Pedalboard = list

# -------- EFFECT BOARDS --------
doom_board = Pedalboard([
    HighpassFilter(85), Distortion(drive_db=30), Gain(gain_db=8), LowpassFilter(7500), Convolution("irs/punk_ir.wav")
])
punk_board = Pedalboard([
    HighpassFilter(90), Distortion(drive_db=28), Gain(gain_db=6), LowpassFilter(7500), Convolution("irs/punk_ir.wav")
])
bossa_board = Pedalboard([
    HighpassFilter(120), Gain(gain_db=1), LowpassFilter(9000), Convolution("irs/bossa_ir.wav")
])

# --- FIX 1: ALIGN GENRE NAMES ---
# Your training script uses 'bossa', so this key MUST be 'bossa'
effect_chains = { 
    "doom": doom_board, 
    "punk": punk_board, 
    "bossa": bossa_board 
}

# -------- FLASK + CONFIG --------
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

SR = 44100
TARGET_SR = 16000
BLOCK = 512
CHANNELS = 1

print("Loading YAMNet...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1') 
print("Loading Classifier...")
clf = joblib.load("genre_classifier_yamnet.pkl") 

processing_state = {
    "current_genre": " ",
    "level": 0.0,
    "live_mode": False,
    "last_file_result": {"genre": " "},
    "collecting": False,
    "locked_genre": None,
    "collect_time_left": 0
}

recognition_buffer = []
COLLECT_DURATION = 8
q = queue.Queue(maxsize=100)

# -------- HELPER --------
def get_yamnet_embedding_vector(audio, sr):
    if sr != TARGET_SR:
        if len(audio) == 0: return None
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # --- FIX 2: THE NOISE GATE ---
    # Calculate volume BEFORE normalization
    rms = np.sqrt(np.mean(audio**2))
    
    # If the signal is just background hum (too quiet), 
    # DO NOT normalize it. Return None so we treat it as 'noise'.
    # Adjust 0.01 if you need it more/less sensitive.
    if rms < 0.01:
        return None

    # NOW it is safe to normalize (because we know it's actually music)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    emb_result = embeddings.numpy()
    if len(emb_result) > 0:
        return np.mean(emb_result, axis=0).reshape(1, -1)
    return None


def processing_thread():
    buffer_len = int(SR * 2.0) 
    buffer = np.zeros(buffer_len, dtype=np.float32)

    while True:
        block = q.get()
        if block is None: break
        if not processing_state["live_mode"]: continue

        buffer = np.roll(buffer, -len(block))
        buffer[-len(block):] = block.flatten()

        if processing_state["collecting"]:
            try:
                analysis_window = buffer[-int(SR * 1.0):]
                feat = get_yamnet_embedding_vector(analysis_window, SR)
                
                if feat is not None:
                    # If sound is loud enough, predict genre
                    genre = clf.predict(feat)[0]
                else:
                    # If sound is too quiet, assume noise
                    genre = "noise"

                processing_state["current_genre"] = genre
                recognition_buffer.append(genre)

            except Exception as e:
                print(f"Live Prediction Error: {e}")

        rms_val = np.sqrt(np.mean(block**2))
        processing_state["level"] = float(rms_val)

Thread(target=processing_thread, daemon=True).start()

# -------- AUDIO STREAM --------
stream = None

def audio_callback(indata, outdata, frames, time_info, status):
    if processing_state["live_mode"]:
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            pass

    if processing_state["locked_genre"] is None:
        outdata[:] = indata
        return

    genre = processing_state["locked_genre"]

    # --- FIX 3: IGNORE NOISE ---
    if genre == "noise":
        # Pass audio through CLEANLY (or keep previous effect)
        outdata[:] = indata
        return

    if genre in effect_chains:
        try:
            processed = effect_chains[genre](indata.copy(), SR)
            outdata[:] = processed
        except:
            outdata[:] = indata
    else:
        outdata[:] = indata

def start_audio():
    global stream
    if stream is not None: return
    stream = sd.Stream(samplerate=SR, blocksize=BLOCK, channels=CHANNELS, latency='low', callback=audio_callback)
    stream.start()
    print("Audio System Started")

# -------- TIMER --------
def finish_collect_timer():
    for t in range(COLLECT_DURATION, 0, -1):
        processing_state["collect_time_left"] = t
        time.sleep(1)
    processing_state["collect_time_left"] = 0
    processing_state["collecting"] = False

    if len(recognition_buffer) > 0:
        # Count votes
        final = max(set(recognition_buffer), key=recognition_buffer.count)
        # If the winner is noise, don't lock it in—maybe keep searching or default to clean
        processing_state["locked_genre"] = final
        print(f"LOCKED IN: {final}")
    else:
        processing_state["locked_genre"] = None

# -------- API ROUTES (Standard) --------
@app.post("/enable_live")
def enable_live():
    processing_state["live_mode"] = True
    processing_state["collecting"] = True
    processing_state["locked_genre"] = None
    recognition_buffer.clear()
    Thread(target=finish_collect_timer, daemon=True).start()
    start_audio()
    return jsonify({"ok": True})

@app.post("/disable_live")
def disable_live():
    processing_state["live_mode"] = False
    processing_state["collecting"] = False
    processing_state["locked_genre"] = None
    processing_state["current_genre"] = " "
    return jsonify(processing_state)

@app.post("/reset_effect")
def reset_effect():
    processing_state["locked_genre"] = None
    processing_state["collecting"] = False
    recognition_buffer.clear()
    return jsonify({"ok": True})

@app.get("/state")
def state():
    return jsonify(processing_state)

@app.get("/")
def index():
    try: return open("templates/index.html").read()
    except: return "Ensure templates/index.html exists"

if __name__ == "__main__":
    start_audio()
    app.run(debug=False, host="0.0.0.0", port=5000)