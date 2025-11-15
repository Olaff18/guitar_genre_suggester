# backend.py
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
import soundfile as sf
from pedalboard import Pedalboard, Distortion, Gain, Reverb, LowpassFilter, HighpassFilter, Convolution, Chorus

# --- EFFECT CHAINS ---

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

# metal_board = Pedalboard([
#     HighpassFilter(140),
#     Gain(gain_db=12),
#     Distortion(drive_db=45),
#     LowpassFilter(9000),
#     Convolution("irs/mesa_4x12.wav")
# ])

# connect genre names → boards
effect_chains = {
    "doom": doom_board,
    "punk": punk_board,
    "bossa_nova": bossa_board,
    # "metal": metal_board
}


# konfig
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

SR = 44100
BLOCK = 2048
CLASSIFY_WINDOW = 1.0
CHANNELS = 1

# laduje model klasyfikatora
clf = joblib.load("genre_classifier.pkl")

# global state
processing_state = {
    "current_genre": " ",
    "level": 0.0,
    "live_mode": False,
    "last_file_result": {"genre" : " " }
}

q = queue.Queue(maxsize=40)

# ekstrakcja cech
def extract_features_block(block, sr=SR):
    y = block.flatten().astype(np.float32)

    if y.size == 0:
        return None
    
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    # staty dla extrakcji cech
    def stats(x):
        return [np.mean(x), np.std(x), np.median(x)]

    centroid = stats(librosa.feature.spectral_centroid(y=y, sr=sr)[0]) # czyli srodek masy spektrum czyli dzwiek jest wysoki czy niski
    rms = stats(librosa.feature.rms(y=y)[0]) # root mean square czyli glosnosc dzwieku czyli jego energia 
    zcr = stats(librosa.feature.zero_crossing_rate(y=y)[0]) # zero crossing rate czyli ilosc zmian znaku w sygnale czyli szumowatosc dzwieku
    bandwidth = stats(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]) # szerokosc pasma czyli jak szerokie sa czestotliwosci w dzwieku

    # MFCC czyli wspolczynniki cepstralne mel czyli jak ludzkie ucho odbiera dzwiek
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    # kontrast spektralny czyli roznice miedzy pasmami czestotliwosci czyli sa dobrze oddzielone
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_means = np.mean(contrast, axis=1)

    # rolloff czyli ksztalt spektrum czyli wysokie czestotliwosci
    roll = stats(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])

    return np.concatenate([
        centroid, rms, zcr, bandwidth,
        mfcc_means, mfcc_stds,
        contrast_means,
        roll
    ])

#thread processing audio blocks
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
        except:
            pass

        rms_val = np.sqrt(np.mean(buffer**2))
        processing_state["level"] = float(rms_val)

thread = Thread(target=processing_thread, daemon=True)
thread.start()

#tu stream audio
stream = None

def audio_callback(indata, outdata, frames, time_info, status):
    if processing_state["live_mode"]:
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            pass

    # outdata[:] = indata  # passthrough monitoring
    genre = processing_state["current_genre"]

    if processing_state["live_mode"] and genre in effect_chains:
        chain = effect_chains[genre]
        try:
            processed = chain(indata.copy(), SR)
            outdata[:] = processed
        except Exception as e:
            print("Effect error:", e)
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

#api routes
@app.post("/enable_live")
def enable_live():
    processing_state["live_mode"] = True
    start_audio()
    return jsonify({"ok": True})

@app.post("/disable_live")
def disable_live():
    processing_state["live_mode"] = False
    processing_state["current_genre"] = " "
    processing_state["level"] = 0.0
    # result = {"genre": " "}
    return jsonify(processing_state)

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
    except Exception as e:
        return jsonify({"error": str(e)}), 500
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
