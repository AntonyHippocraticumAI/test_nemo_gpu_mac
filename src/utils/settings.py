import torch

# Name of the audio file
# AUDIO_PATH = "/Users/antonandreev/python_prog/test_nemo_cpu_mac/audio_samples/recording_16000hz.wav"
# AUDIO_PATH = "audio_samples/ravi_inva.wav"
AUDIO_PATH = "audio_samples/recording_16000hz.wav"

# Whether to enable music removal from speech, helps increase diarization quality but uses alot of ram
enable_stemming = True

# (choose from 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large')
whisper_model_name = "large-v3"

compute_type = "int8"
# or run on GPU with INT8
# compute_type = "int8_float16"
# or run on CPU with INT8
# compute_type = "int8"

# replaces numerical digits with their pronounciation, increases diarization accuracy
suppress_numerals = True

batch_size = 8
beams_size = 5

language = None  # autodetect language

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

WHISPER_MODEL_NAME = "large-v3"
VAD_MODEL_PATH = "nemo_models/vad/vad_marblenet.nemo"   # або локальний .nemo файл
EMBED_MODEL_PATH = "nemo_models/speaker_embeddings/titanet-l.nemo"

import torch 
import numpy
import nemo.collections.asr as nemo_asr
print('NeMo loaded!')
print(numpy.__version__)
