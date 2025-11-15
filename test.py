# test_effects.py
import soundfile as sf
from pedalboard import Pedalboard, Distortion, Gain, Reverb, LowpassFilter

# 1. load a clean guitar WAV
audio, sr = sf.read("training/bach/untitled.wav")

# 2. build your pedal chain
board = Pedalboard([
    Distortion(drive_db=30),
    Gain(gain_db=10),
    LowpassFilter(8000),
    Reverb(room_size=0.7, wet_level=0.5)
])

# 3. run clean guitar → effects
processed = board(audio, sr)

# 4. save new file
sf.write("processed_guitar.wav", processed, sr)
