from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Distortion, Gain, Reverb, Compressor
from pedalboard.io import AudioFile

# 1. Definiujemy Twój Pedalboard (np. wersja DOOM bez IR)
doom_board = Pedalboard([
    HighpassFilter(40),       # Lekkie podcięcie dołu
    Distortion(drive_db=35),  # Mocny przester
    Gain(gain_db=6),          # Boost głośności
    LowpassFilter(4500),      # Symulacja ciemnej kolumny (zamiast IR)
])

# PUNK:
punk_board = Pedalboard([
    HighpassFilter(100),
    Distortion(drive_db=24),
    Gain(gain_db=4),
    LowpassFilter(6500),
])

# BOSSA: Czysto, ciepło, z wyrównaną dynamiką
bossa_board = Pedalboard([
    # Kompresor jest tu KLUCZOWY. Wyrównuje ciche pociągnięcia palcami.
    Compressor(threshold_db=-15, ratio=3),
    # Pozwalamy wybrzmieć ciepłym dołom (body gitary)
    HighpassFilter(60),
    # minimalny drive czasem dodaje ciepła lampy, ale bardzo mało
    Distortion(drive_db=5), 
    Gain(gain_db=2),
    LowpassFilter(8000),
    
    # Opcjonalnie Reverb (Bossa lubi przestrzeń, ale nie "studnię")
    Reverb(room_size=0.5, wet_level=0.3)
])

# Nazwy plików
input_file = "bossa_nova_1.wav"  # Twój plik wejściowy
output_file = "bossaeff1.wav"  # Plik wynikowy

print(f"Przetwarzanie pliku: {input_file}...")

# 2. Otwieramy plik do odczytu
with AudioFile(input_file) as f:
    # Wczytujemy całe audio do pamięci
    audio = f.read(f.frames)
    # Pobieramy częstotliwość próbkowania (np. 44100 Hz lub 48000 Hz)
    samplerate = f.samplerate

# 3. Nakładamy efekty
# To jest kluczowy moment - biblioteka przetwarza tablicę numpy
effected_audio = bossa_board(audio, samplerate)

# 4. Zapisujemy przetworzony plik
# Otwieramy plik do zapisu ('w'), podajemy samplerate i liczbę kanałów
with AudioFile(output_file, 'w', samplerate, effected_audio.shape[0]) as f:
    f.write(effected_audio)

print(f"Gotowe! Zapisano jako: {output_file}")