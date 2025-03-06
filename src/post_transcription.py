import os
import re

from services.model_manager.vad_manager import preload_vad_model
from services.model_manager.whisper_manager import preload_models
from utils.result_utils import generate_readable_filename, split_by_words
from utils.settings import AUDIO_PATH, batch_size, beams_size
from utils.logging_utils import get_logger

logger = get_logger(__name__)

#-----------------------------------------------------------------------------------------------------------------------

whisper_model_manager = preload_models()
vad_model_manager = preload_vad_model()
logger.info("Loaded VAD and WHISPER models")

#-----------------------------------------------------------------------------------------------------------------------

audio, sr = vad_model_manager.get_audio_time_series(AUDIO_PATH)
logger.info(f"AUDIO: {audio}")

params_vad = vad_model_manager.get_params_for_vad_model(audio, sr, window_length=0.3, onset_thresh=0.4, min_speech_dur=0.3)
logger.info(f"PARAMS VAD: {params_vad}")

segments = vad_model_manager.get_speech_segments(**params_vad)
logger.info(f"SEGMENTS: {segments}")

cleaned_audio = vad_model_manager.get_ndarray_of_segments(audio, segments, sr)
logger.info(f"CLEANED AUDIO: {cleaned_audio}")

full_transcript, info = whisper_model_manager.transcribe_with_vad(cleaned_audio, batch_flag=True)
logger.info(f"FINAL TRANSCRIPT: {full_transcript}")


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "segmentations_and_transcriptions/")
RESULT_FILE_NAME = generate_readable_filename(RESULT_DIR)
os.makedirs(os.path.dirname(RESULT_FILE_NAME), exist_ok=True)

with open(f"{RESULT_FILE_NAME}.txt", "w", encoding="utf-8") as f:
    f.write("PARAMS FOR VAD:\n")
    for i in params_vad:
        f.write(f"{i}: {params_vad[i]}\n")

    f.write("\n")

    f.write(f"VAD SEGMENTS: {segments}\n\n")

    f.write("PARAMS FOR WHISPER:\n")
    f.write(f"BATCH SIZE: {batch_size}; BEAM_SIZE: {beams_size}\n\n")
    f.write(f"TRANSCRIPT:\n\n")
    f.write(f"{full_transcript}\n\n")
    f.write(f"{split_by_words(full_transcript)}\n")

