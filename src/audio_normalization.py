import soundfile as sf
import pyloudnorm as pyln

data, rate = sf.read("audio_samples/M_0234_9y9m_1.wav") # load audio (with shape (samples, channels))
meter = pyln.Meter(rate) # create BS.1770 meter
loudness = meter.integrated_loudness(data) # measure loudness


peak_normalized_audio = pyln.normalize.peak(data, -1.0)

# measure the loudness first 
meter = pyln.Meter(rate) # create BS.1770 meter
loudness = meter.integrated_loudness(data)

# loudness normalize audio to -12 dB LUFS
loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -12.0)
print(loudness_normalized_audio)

# Зберігаємо файл після пікової нормалізації
sf.write("normalized_peak.wav", peak_normalized_audio, rate)

# Зберігаємо файл після нормалізації за гучністю (LUFS)
sf.write("normalized_loudness.wav", loudness_normalized_audio, rate)
