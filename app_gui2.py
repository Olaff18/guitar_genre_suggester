import os
import queue
import threading
import numpy as np
import sounddevice as sd
import joblib
import scipy.signal
import librosa
import PySimpleGUI as sg
from nam_backend import NAMWrapper
import scipy.io.wavfile as wavfile

# ------------------------
# CONFIG
# ------------------------
SR = 44100
BLOCK = 2048
CLASSIFY_WINDOW = 1.0  # seconds for feature extraction
CHANNELS = 1

MODELS_DIR = "models"
IRS_DIR = "irs"

MODEL_FILES = {
    "metal": os.path.join(MODELS_DIR, "metal.nam"),
    "doom": os.path.join(MODELS_DIR, "doom.nam"),
    "shoegaze": os.path.join(MODELS_DIR, "shoegaze.nam"),
    "punk": os.path.join(MODELS_DIR, "punk.nam"),
}

IR_FILES = {
    "metal": os.path.join(IRS_DIR, "metal_ir.wav"),
    "doom": os.path.join(IRS_DIR, "doom_ir.wav"),
    "shoegaze": os.path.join(IRS_DIR, "shoegaze_ir.wav"),
    "punk": os.path.join(IRS_DIR, "punk_ir.wav"),
}

CLASSIFIER_PATH = "genre_classifier.pkl"
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError("Train classifier first (expect genre_classifier.pkl in project root).")
clf = joblib.load(CLASSIFIER_PATH)

# ------------------------
# AUDIO QUEUE
# ------------------------
audio_queue = queue.Queue(maxsize=20)

# ------------------------
# NAM WRAPPER + IR LOADER
# ------------------------
nam_wrapper = NAMWrapper()
current_ir = None

def load_ir(path):
    if not os.path.exists(path):
        print(f"IR file not found: {path}")
        return None
    try:
        sr, data = wavfile.read(path)
        # if data is int (scalar), make it array
        if np.isscalar(data):
            data = np.array([data], dtype=np.float32)
        else:
            data = data.astype(np.float32)
        # resample if needed
        if sr != SR:
            data = librosa.resample(data, orig_sr=sr, target_sr=SR)
        # mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # normalize
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        return data
    except Exception as e:
        print(f"Failed to load IR {path}: {e}")
        return None

# ------------------------
# FEATURE EXTRACTION
# ------------------------
def extract_features_block(block):
    y = block.flatten()
    if len(y) < 256:
        y = np.pad(y, (0, 256 - len(y)))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=SR))
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=SR))
    return [centroid, rms, zcr, bandwidth]

# ------------------------
# FFT CONVOLUTION (IR)
# ------------------------
def fft_convolve_block(signal_block, ir):
    if ir is None:
        return signal_block
    out = scipy.signal.fftconvolve(signal_block, ir)[:len(signal_block)]
    if np.max(np.abs(out)) > 0:
        out /= max(1.0, np.max(np.abs(out)))
    return out.astype(np.float32)

# ------------------------
# PROCESSING THREAD
# ------------------------
processing_state = {
    "buffer": np.zeros(int(CLASSIFY_WINDOW * SR), dtype=np.float32),
    "current_genre": None
}

def processing_thread(out_queue):
    global current_ir
    while True:
        block = audio_queue.get()
        if block is None:
            break
        # update sliding window buffer
        b = processing_state["buffer"]
        b = np.concatenate([b[len(block):], block.flatten()])
        processing_state["buffer"] = b
        # extract features & classify
        feat = extract_features_block(b)
        genre = clf.predict([feat])[0]
        if genre != processing_state["current_genre"]:
            processing_state["current_genre"] = genre
            # load NAM + IR
            model_path = MODEL_FILES.get(genre)
            ir_path = IR_FILES.get(genre)
            if model_path and os.path.exists(model_path):
                nam_wrapper.load(model_path, "")
            current_ir = load_ir(ir_path)
            print(f"Switched to {genre}")
        # process NAM
        processed = nam_wrapper.process(block.flatten())
        # apply IR
        out_block = fft_convolve_block(processed, current_ir)
        out_queue.put(out_block)

# ------------------------
# AUDIO CALLBACK
# ------------------------
def audio_callback(indata, outdata, frames, time_info, status):
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass
    try:
        block = out_queue.get_nowait()
        outdata[:] = block.reshape(outdata.shape)
    except queue.Empty:
        outdata[:] = indata

# ------------------------
# GUI
# ------------------------
def main():
    sg.theme("DarkBlue3")
    layout = [
        [sg.Text("Guitar Genre Tone Switcher", font=("Any", 18))],
        [sg.Text("Current genre:"), sg.Text("—", key="-GENRE-", size=(12,1), font=("Any", 14))],
        [sg.ProgressBar(max_value=100, orientation='h', size=(40, 20), key='-LEVEL-')],
        [sg.Button("Start", key="-START-"), sg.Button("Stop", key="-STOP-"), sg.Button("Quit")],
        [sg.Text("Status:"), sg.Text("", key="-STATUS-", size=(40,1))]
    ]
    window = sg.Window("ToneSwitcher", layout)

    global out_queue
    out_queue = queue.Queue(maxsize=50)
    thread = threading.Thread(target=processing_thread, args=(out_queue,), daemon=True)
    thread.start()

    stream = None
    running = False

    while True:
        event, values = window.read(timeout=50)
        if event in (sg.WIN_CLOSED, "Quit"):
            break
        if event == "-START-" and not running:
            try:
                stream = sd.Stream(samplerate=SR, blocksize=BLOCK, channels=1, callback=audio_callback)
                stream.start()
                running = True
                window['-STATUS-'].update("Running")
            except Exception as e:
                window['-STATUS-'].update(f"Audio start error: {e}")
        if event == "-STOP-" and running:
            stream.stop()
            stream.close()
            stream = None
            running = False
            window['-STATUS-'].update("Stopped")
        # update genre in GUI
        window['-GENRE-'].update(processing_state["current_genre"] or "—")
        # update level
        rms = np.sqrt(np.mean(processing_state["buffer"]**2))
        window['-LEVEL-'].update(int(min(100, rms*200)))

    audio_queue.put(None)
    out_queue.put(None)
    window.close()

if __name__ == "__main__":
    main()
