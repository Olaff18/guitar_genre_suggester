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

# -------- EFFECT BOARDS --------

doom_board = Pedalboard([
    HighpassFilter(70),
    Distortion(drive_db=40),
    Gain(gain_db=15),
    LowpassFilter(3000),
    Convolution("irs/doom_ir.wav")
])

punk_board = Pedalboard([
    HighpassFilter(120),
    Distortion(drive_db=30),
    Gain(gain_db=5),
    Convolution("irs/punk_ir.wav")
])

bossa_board = Pedalboard([
    Gain(gain_db=8),
    Chorus(rate_hz=0.6, depth=1.0),
    Reverb(room_size=0.9, damping=0.2, wet_level=0.65, dry_level=0.4),
    LowpassFilter(8000),
    Convolution("irs/shoegaze_ir.wav")
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
BLOCK = 2048
CLASSIFY_WINDOW = 1.0
CHANNELS = 1

clf = joblib.load("genre_classifier.pkl")

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
    if y.size == 0:
        return None
    
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    def stats(x):
        return [np.mean(x), np.std(x), np.median(x)]

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
    buffer = np.zeros(int(SR * CLASSIFY_WINDOW), dtype=np.float32)

    while True:
        block = q.get()
        if block is None:
            break

        if not processing_state["live_mode"]:
            continue

        buffer = np.concatenate([buffer[len(block):], block.flatten()])
        feat = extract_features_block(buffer)

        try:
            genre = clf.predict([feat])[0]
            processing_state["current_genre"] = genre

            if processing_state["collecting"]:
                recognition_buffer.append(genre)

        except:
            pass

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
