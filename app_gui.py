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




SR = 44100
BLOCK = 2048             # block size for callback
CLASSIFY_WINDOW = 1.0    # seconds for feature extraction
CHANNELS = 1


# load classifier
CLASSIFIER_PATH = "genre_classifier.pkl"
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError("Train classifier first (training/train_classifier.py). Expect genre_classifier.pkl in project root.")
clf = joblib.load(CLASSIFIER_PATH)

# audio queue to pass data from callback to processing thread
q = queue.Queue(maxsize=20)

# lightweight feature extraction
def extract_features_block(block, sr=SR):
    y = block.flatten().astype(np.float32)

    if y.size == 0:
        return None

    # Ensure min length to avoid librosa crashes
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    # -----------------------------
    # BASIC FEATURES (3 stats each)
    # -----------------------------
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # take multiple statistics, not just mean
    def stats(x):
        return [np.mean(x), np.std(x), np.median(x)]

    feat_centroid = stats(centroid)
    feat_rms = stats(rms)
    feat_zcr = stats(zcr)
    feat_bandwidth = stats(bandwidth)

    # -----------------------------
    # MFCCs (13 coefficients)
    # -----------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    # -----------------------------
    # Spectral Contrast (7 bands)
    # -----------------------------
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_means = np.mean(contrast, axis=1)

    # -----------------------------
    # Rolloff (shape of spectrum)
    # -----------------------------
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feat_rolloff = stats(rolloff)

    # -----------------------------
    # FINAL FEATURE VECTOR
    # -----------------------------
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
            
            print(f"Switched to {genre}")
        

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