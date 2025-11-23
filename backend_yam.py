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
import tensorflow_hub as hub # <--- NEW: Required for YAMNet

# Check for Pedalboard
try:
    from pedalboard import Pedalboard, Distortion, Gain, Reverb, LowpassFilter, HighpassFilter, Convolution
except ImportError:
    print("Warning: Pedalboard not found. Effects disabled.")
    Pedalboard = list

# -------- EFFECT BOARDS --------
# (Your effect settings are perfect, keeping them as is)
doom_board = Pedalboard([
    HighpassFilter(85), Distortion(drive_db=30), Gain(gain_db=8), LowpassFilter(7500), Convolution("irs/punk_ir.wav")
])
punk_board = Pedalboard([
    HighpassFilter(90), Distortion(drive_db=28), Gain(gain_db=6), LowpassFilter(7500), Convolution("irs/punk_ir.wav")
])
bossa_board = Pedalboard([
    HighpassFilter(120), Gain(gain_db=1), LowpassFilter(9000), Convolution("irs/bossa_ir.wav")
])

effect_chains = { "doom": doom_board, "punk": punk_board, "bossa_nova": bossa_board }

# -------- FLASK + CONFIG --------
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

SR = 44100
TARGET_SR = 16000 # YAMNet requires 16k
BLOCK = 512
CHANNELS = 1

# --- LOAD MODELS ---
print("Loading YAMNet... (This takes a moment)")
# Load YAMNet once globally so we don't reload it every second
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1') 

print("Loading Classifier...")
# FIXED: Load the correct YAMNet-trained classifier
clf = joblib.load("genre_classifier_yamnet.pkl") 

# global state
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

# -------- HELPER: EXTRACT YAMNET FEATURES --------
# We use this for BOTH live audio and uploaded files
def get_yamnet_embedding_vector(audio, sr):
    # 1. Resample to 16kHz
    if sr != TARGET_SR:
        # Safety check for empty audio
        if len(audio) == 0: return None
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # 2. Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # 3. YAMNet Inference
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    emb_result = embeddings.numpy()
    
    if len(emb_result) > 0:
        # Return the mean embedding (1, 1024)
        return np.mean(emb_result, axis=0).reshape(1, -1)
    
    return None


# -------- BACKGROUND CLASSIFIER THREAD --------
def processing_thread():
    # Keep a buffer of 2 seconds
    buffer_len = int(SR * 2.0) 
    buffer = np.zeros(buffer_len, dtype=np.float32)

    while True:
        block = q.get()
        if block is None: break
        if not processing_state["live_mode"]: continue

        # Roll buffer (Rolling Window)
        buffer = np.roll(buffer, -len(block))
        buffer[-len(block):] = block.flatten()

        if processing_state["collecting"]:
            try:
                # 1. Grab last 1.0 second for analysis
                analysis_window = buffer[-int(SR * 1.0):]

                # 2. Use the helper function
                feat = get_yamnet_embedding_vector(analysis_window, SR)
                
                if feat is not None:
                    # 3. Predict using Random Forest
                    genre = clf.predict(feat)[0]
                    
                    processing_state["current_genre"] = genre
                    recognition_buffer.append(genre)
                    # print(f"YAMNet thinks: {genre}") # Debug

            except Exception as e:
                print(f"Live Prediction Error: {e}")

        # Level meter
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
    if genre == "noise":
        # Do not change the effect. 
        # Just keep passing audio through the PREVIOUS effect (or clean).
        # For simplicity here, we just pass clean audio, or you can keep the last effect active.
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
    
    # Try ASIO for Windows if available, otherwise default
    # device_config = 'ASIO4ALL v2' 
    stream = sd.Stream(
        samplerate=SR,
        blocksize=BLOCK,
        channels=CHANNELS,
        latency='low',
        callback=audio_callback
    )
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
        processing_state["locked_genre"] = max(set(recognition_buffer), key=recognition_buffer.count)
        print(f"LOCKED IN: {processing_state['locked_genre']}")
    else:
        processing_state["locked_genre"] = None

# -------- API ROUTES --------

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

@app.post("/classify_file")
def classify_file():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    f = request.files["audio"]
    tmp = tempfile.mktemp(suffix=f.filename)
    f.save(tmp)

    try:
        # Load File
        y, sr = librosa.load(tmp, sr=SR, mono=True)
        
        # FIXED: Use YAMNet helper instead of old manual features
        feat = get_yamnet_embedding_vector(y, sr)
        
        if feat is not None:
            genre = clf.predict(feat)[0]
        else:
            genre = "Unknown"

        result = {"genre": genre}
        processing_state["last_file_result"] = result
        return jsonify(result)

    finally:
        try: os.remove(tmp)
        except: pass

@app.get("/state")
def state():
    return jsonify(processing_state)

@app.get("/")
def index():
    try:
        return open("templates/index.html").read()
    except:
        return "Ensure templates/index.html exists"

if __name__ == "__main__":
    start_audio()
    app.run(debug=False, host="0.0.0.0", port=5000)