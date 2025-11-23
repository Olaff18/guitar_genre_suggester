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
from pedalboard import Pedalboard, Distortion, Gain, Reverb, LowpassFilter, HighpassFilter, Convolution, Chorus

CLASSIFY_WINDOW = 3.0
# -------- EFFECT BOARDS --------

doom_board = Pedalboard([
    HighpassFilter(85),                   # tight low end
    Distortion(drive_db=30),              # Marshall-ish crunch
    Gain(gain_db=8),                      # boost but not too much
    LowpassFilter(7500),                  # punk has bite but not fizz
    Convolution("irs/punk_ir.wav")
])


punk_board = Pedalboard([
    HighpassFilter(90),                   # tight low end
    Distortion(drive_db=28),              # Marshall-ish crunch
    Gain(gain_db=6),                      # boost but not too much
    LowpassFilter(7500),                  # punk has bite but not fizz
    Convolution("irs/punk_ir.wav")
])


bossa_board = Pedalboard([
    HighpassFilter(120),                 
    Gain(gain_db=1),                      # small boost = warmth
    # Chorus(rate_hz=1.0, depth=0.4),       # soft jazz chorus (not 80s)
    # Reverb(room_size=0.85, damping=0.25, wet_level=0.55),
    LowpassFilter(9000),                  # warm mellow tone
    Convolution("irs/bossa_ir.wav")      # warm clean cab IR
])


effect_chains = {
    "doom": doom_board,
    "punk": punk_board,
    "bossa_nova": bossa_board,
}


# -------- FLASK + CONFIG --------

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

SR = 44100
BLOCK = 512
CLASSIFY_WINDOW = 3.0
CHANNELS = 1

clf = joblib.load("genre_classifier.pkl")
scaler = joblib.load("genre_scaler.pkl")

# global state
processing_state = {
    "current_genre": " ",
    "level": 0.0,
    "live_mode": False,
    "last_file_result": {"genre": " "},

    # new:
    "collecting": False,         # during the 5-second analysis
    "locked_genre": None,        # selected effect after analysis
    "collect_time_left": 0       # countdown for UI
}

recognition_buffer = []
COLLECT_DURATION = 8

q = queue.Queue(maxsize=40)

# -------- FEATURE EXTRACTION --------
def extract_features_block(block, sr=SR):

    y = block.flatten().astype(np.float32)

    if np.max(np.abs(y)) < 0.005:  # too silent
        return None 
    
    y = librosa.util.normalize(y) # normalizacja glośności zeby nie bylo za cicho lub za glosno
    

    if y.size == 0:
        return None
    
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    def stats(x):
        return [np.mean(x), np.std(x)]

    centroid = stats(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    rms = stats(librosa.feature.rms(y=y)[0])
    zcr = stats(librosa.feature.zero_crossing_rate(y=y)[0])
    bandwidth = stats(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_means = np.mean(contrast, axis=1)

    roll = stats(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])

    return np.concatenate([
        centroid, rms, zcr, bandwidth,
        mfcc_means, mfcc_stds,
        contrast_means,
        roll
    ])


# -------- BACKGROUND CLASSIFIER THREAD --------
def processing_thread():
    buffer_length = int(SR * CLASSIFY_WINDOW) # Holds 3 seconds
    buffer = np.zeros(buffer_length, dtype=np.float32)

    while True:
        block = q.get()
        if block is None:
            break

        if not processing_state["live_mode"]:
            continue

        # ahift the array to the left, add new block on the right
        # this allows us to always analyze the "last 3 seconds"
        buffer = np.roll(buffer, -len(block))
        buffer[-len(block):] = block.flatten()       

        # Only run prediction logic if we are in the "Collecting" phase
        if processing_state["collecting"]:
            
            # We only predict periodically to save CPU, or every block?
            # Every block is fine with Random Forest
            feat = extract_features_block(buffer)

            if feat is not None:
                try:
                    # ---  APPLY SCALER ---
                    # Wwe must translate the live features using the saved scaler
                    feat_reshaped = feat.reshape(1, -1)
                    feat_scaled = scaler.transform(feat_reshaped)
                    
                    genre = clf.predict(feat_scaled)[0]
                    processing_state["current_genre"] = genre
                    recognition_buffer.append(genre)
                except Exception as e:
                    print(f"Pred Error: {e}")

        rms_val = np.sqrt(np.mean(buffer**2))
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

    # If no locked effect → clean
    if processing_state["locked_genre"] is None:
        outdata[:] = indata
        return

    genre = processing_state["locked_genre"]

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
    if stream is not None:
        return
    stream = sd.Stream(
        samplerate=SR,
        # device = 'ASIO4ALL',
        blocksize=BLOCK,
        channels=CHANNELS,
        callback=audio_callback
    )
    stream.start()

# -------- TIMER TO END ANALYSIS --------
def finish_collect_timer():
    for t in range(COLLECT_DURATION, 0, -1):
        processing_state["collect_time_left"] = t
        time.sleep(1)

    processing_state["collect_time_left"] = 0
    processing_state["collecting"] = False

    if len(recognition_buffer) > 0:
        processing_state["locked_genre"] = max(set(recognition_buffer), key=recognition_buffer.count)
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
    processing_state["level"] = 0.0
    return jsonify(processing_state)


@app.post("/reset_effect")
def reset_effect():
    processing_state["locked_genre"] = None
    processing_state["collecting"] = False
    recognition_buffer.clear()
    processing_state["current_genre"] = " "
    return jsonify({"ok": True})


@app.post("/classify_file")
def classify_file():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    f = request.files["audio"]
    tmp = tempfile.mktemp(suffix=f.filename)
    f.save(tmp)

    try:
        y, sr = librosa.load(tmp, sr=SR, mono=True)
        feat = extract_features_block(y)
        genre = clf.predict([feat])[0]

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
    return open("templates/index.html").read()


if __name__ == "__main__":
    start_audio()
    app.run(debug=False, host="0.0.0.0", port=5000)
