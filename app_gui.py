import os
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import joblib
import scipy.signal
import PySimpleGUI  as sg
import librosa

# probujemy zimportowac nam model
try:
    # na razie nie odpala nam
    from neural_amp_modeler import NeuralAmpModel
    NAM_AVAILABLE = True
except Exception:
    NAM_AVAILABLE = False
    print("NAM runtime not found. NAM processing will be bypassed.")

SR = 44100
BLOCK = 2048             # block size for callback
CLASSIFY_WINDOW = 1.0    # seconds for feature extraction
CHANNELS = 1

MODELS_DIR = "models"
IRS_DIR = "irs"
MODEL_FILES = {
    "metal": os.path.join(MODELS_DIR, "metal.nam"),
    "shoegaze": os.path.join(MODELS_DIR, "shoegaze.nam"),
    "punk": os.path.join(MODELS_DIR, "punk.nam"),
    "doom": os.path.join(MODELS_DIR, "doom.nam"),
}
IR_FILES = {
    "metal": os.path.join(IRS_DIR, "metal_ir.wav"),
    "shoegaze": os.path.join(IRS_DIR, "shoegaze_ir.wav"),
    "punk": os.path.join(IRS_DIR, "punk_ir.wav"),
    "doom": os.path.join(IRS_DIR, "doom_ir.wav"),
}

# load classifier
CLASSIFIER_PATH = "genre_classifier.pkl"
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError("Train classifier first (training/train_classifier.py). Expect genre_classifier.pkl in project root.")
clf = joblib.load(CLASSIFIER_PATH)

# audio queue to pass data from callback to processing thread
q = queue.Queue(maxsize=20)

# lightweight feature extraction
def extract_features_block(block, sr=SR):
    y = block.flatten()
    if len(y) < 256:
        # pad
        y = np.pad(y, (0, max(0, 256-len(y))))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    return [centroid, rms, zcr, bandwidth]

# simple IR loader (loads wav, returns normalized numpy)
import scipy.io.wavfile as wavfile
def load_ir(path):
    if not os.path.exists(path):
        return None
    sr, data = wavfile.read(path)
    # resample if needed
    if sr != SR:
        data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=SR)
    # mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    # normalize
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    return data

# setup initial IRs dict
irs = {k: load_ir(v) for k, v in IR_FILES.items()}

# NAM model holder ; maly wrapper
class NAMWrapper:
    def __init__(self):
        self.model = None
    def load(self, path):
        if not NAM_AVAILABLE:
            print("NAM not available; skipping load")
            self.model = None
            return
        
        try:
            self.model = NeuralAmpModel.load(path)
            print("Loaded NAM model", path)
        except Exception as e:
            print("Error loading NAM model:", e)
            self.model = None
    def process(self, audio_block):
        if self.model is None:
            return audio_block
        
        try:
            out = self.model(audio_block.astype(np.float32))
            return out
        except Exception as e:
            print("NAM processing failed:", e)
            return audio_block

nam_wrapper = NAMWrapper()

# convolution helper using FFT-based overlap-add 
def fft_convolve_block(signal_block, ir):
    if ir is None:
        return signal_block
    # simple direct convolution for demonstration 
    out = scipy.signal.fftconvolve(signal_block, ir)[:len(signal_block)]
    # normalize if needed
    if np.max(np.abs(out)) > 0:
        out = out / max(1.0, np.max(np.abs(out)))
    return out.astype(np.float32)

# procesujacy thread: consume buffers, detect genre (every N blocks), switch model + IR, process and send out via out_stream buffer
processing_state = {
    "buffer": np.zeros(int(CLASSIFY_WINDOW * SR), dtype=np.float32),
    "current_genre": None,
    "current_ir": None,
    "current_model": None
}

def processing_thread(out_queue):
    while True:
        block = q.get()
        if block is None:
            break
        # append to sliding window buffer
        b = processing_state["buffer"]
        b = np.concatenate([b[len(block):], block.flatten()])
        processing_state["buffer"] = b
        # extract features every time window is full (we're always filling it)
        feat = extract_features_block(b, SR)
        genre = clf.predict([feat])[0]
        if genre != processing_state["current_genre"]:
            processing_state["current_genre"] = genre
            # load NAM and IR
            model_path = MODEL_FILES.get(genre)
            ir_path = IR_FILES.get(genre)
            if model_path and os.path.exists(model_path):
                nam_wrapper.load(model_path)
            else:
                print("NAM model missing for", genre)
                nam_wrapper.load(None)
            ir = irs.get(genre)
            processing_state["current_ir"] = ir
            print(f"Switched to {genre}")
        # process: NAM -> IR
        # run NAM on the block
        processed = nam_wrapper.process(block.flatten())
        # if NAM returned shorter/longer arrays, ensure same length
        if processed is None or len(processed) != len(block.flatten()):
            processed = block.flatten()
        # apply IR via convolution
        out_block = fft_convolve_block(processed, processing_state["current_ir"])
        # push to out_queue for callback to read (we use a simple queue)
        out_queue.put(out_block.astype(np.float32))

# GUI + audio callback logic
def main():
    # GUI layout
    sg.theme("DarkBlue3")
    layout = [
        [sg.Text("Guitar Genre Tone Switcher", font=("Any", 18))],
        [sg.Text("Current genre:"), sg.Text("", key="-GENRE-", size=(12,1), font=("Any", 14))],
        [sg.ProgressBar(max_value=100, orientation='h', size=(40, 20), key='-LEVEL-')],
        [sg.Button("Start", key="-START-"), sg.Button("Stop", key="-STOP-"), sg.Button("Quit")],
        [sg.Text("Status:"), sg.Text("", key="-STATUS-", size=(40,1))]
    ]
    window = sg.Window("ToneSwitcher", layout)

    out_queue = queue.Queue(maxsize=50)
    proc_thread = threading.Thread(target=processing_thread, args=(out_queue,), daemon=True)
    proc_thread.start()

    # audio callback (input -> push to q; read processed output from out_queue)
    def sd_callback(indata, outdata, frames, time_info, status):
        # place incoming block into processing queue (non-blocking)
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            pass
        # try to read processed block for playback
        try:
            block = out_queue.get_nowait()
            outdata[:] = block.reshape(outdata.shape)
        except queue.Empty:
            # pass input as output if no processed data yet
            outdata[:] = indata

    stream = None
    running = False

    while True:
        event, values = window.read(timeout=50)
        if event == sg.WIN_CLOSED or event == "Quit":
            break
        if event == "-START-" and not running:
            try:
                stream = sd.Stream(samplerate=SR, blocksize=BLOCK, channels=CHANNELS, callback=sd_callback)
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
        # update displayed genre
        window['-GENRE-'].update(processing_state["current_genre"] or "—")
        # optional: update level from buffer RMS
        rms = np.sqrt(np.mean(processing_state["buffer"]**2))
        level = int(min(100, rms * 200))  # scaled for progressbar
        window['-LEVEL-'].update(level)

    # cleanup
    q.put(None)
    out_queue.put(None)
    window.close()

if __name__ == "__main__":
    main()
