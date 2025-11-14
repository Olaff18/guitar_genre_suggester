from pedalboard import VST3Plugin
try:
    p = VST3Plugin(r"C:\Program Files\Common Files\VST3\NeuralAmpModeler.vst3")
    print("Loaded OK")
except Exception as e:
    print("ERROR:", e)