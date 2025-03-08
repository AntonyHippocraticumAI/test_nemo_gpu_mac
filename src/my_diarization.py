import os
import librosa
from nemo.collections.asr.models import NeuralDiarizer

from services.model_manager.vad_manager import preload_vad_model
from services.model_manager.whisper_manager import preload_models
from utils.d_utils import DiarizationService
from utils.manifest import create_manifest
from utils.config_models import create_config_diarisation
from utils.rttm_utils import RttmUtils
from utils.settings import AUDIO_PATH
from utils.logging_utils import get_logger


logger = get_logger(__name__)
FORCE_SPEAKERS = None

ROOT_DIR = "./nemo_diar_temp"
os.makedirs(ROOT_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(ROOT_DIR, "input_manifest.json")
OUT_DIR = os.path.join(ROOT_DIR, "diar_output")


def main():
    ##########################################################################
    whisper_model_manager = preload_models()
    # vad_model_manager = preload_vad_model()
    logger.info("Loaded VAD and WHISPER models")
    ##########################################################################
    # audio_without_vad, sr = vad_model_manager.get_audio_time_series(AUDIO_PATH)
    # logger.info(f"AUDIO: {audio_without_vad}")

    # params_vad = vad_model_manager.get_params_for_vad_model(audio_without_vad, sr, window_length=0.2, onset_thresh=0.5, min_speech_dur=0.2)
    # logger.info(f"PARAMS VAD: {params_vad}")

    # segments = vad_model_manager.get_speech_segments(**params_vad)
    # logger.info(f"SEGMENTS: {segments}")

    # audio_data = vad_model_manager.get_ndarray_of_segments(audio_without_vad, segments, sr)
    # logger.info(f"CLEANED AUDIO: {audio_data}")


    # from scipy.io.wavfile import write
    # import numpy as np

    # if audio_data.dtype != np.int16:
    #     audio_data = audio_data / np.max(np.abs(audio_data))
    #     audio_data = np.int16(audio_data * 32767)

    # write("output.wav", 16000, audio_data)

    ##########################################################################


    ##########################################################################
    # A) TRANSCRIPTION WHISPER
    ##########################################################################
    audio, sr = librosa.load(AUDIO_PATH, sr=16000)
    logger.info(f"Loaded audio {AUDIO_PATH}, sr={sr}, duration={len(audio)/sr:.1f}s")

    filename = os.path.splitext(os.path.basename(AUDIO_PATH))[0]

    print("HEEEEEEEEEEEERE", filename)
    

    logger.info("=== WHISPER STEP ===")
    ##########################################################################
    # A) TRANSCRIPTION WHISPER
    ##########################################################################

    segments_gen, _ = whisper_model_manager.transcribe_for_diarisation(audio)
    logger.info(segments_gen)
    whisper_segments = whisper_model_manager.produced_segments(segments_gen)

    for segment_transcription in whisper_segments:
        logger.info(segment_transcription)

    logger.info(f"Whisper produced {len(whisper_segments)} segments.")
    logger.info("=== WHISPER STEP FINISHED ===")

    ##########################################################################
    # B) NEMO DIARIZATION (MarbleNet -> Segmentation -> TitaNet -> Clustering -> MSDD)
    ##########################################################################

    # 1) Create manifest (JSON) for one audio
    create_manifest(AUDIO_PATH, MANIFEST_PATH, FORCE_SPEAKERS)

    # 2) Create config for NeuralDiarizer
    #    enable_msdd=True -> Use MSDD (Multi-scale diarization decoder)
    cfg = create_config_diarisation(MANIFEST_PATH, OUT_DIR)

    # 3) Start NeMo Diarizer
    logger.info("=== NeMo DIARIZATION (with MSDD) ===")
    diar_model = NeuralDiarizer(cfg=cfg)
    diar_model.diarize()

    ##########################################################################
    # C) Read RTTM and merge with Whisper segments
    ##########################################################################

    # Format: SPEAKER <audio_name> 1 <start_time> <duration> <..> <..> <speaker_label>
    rttm_utils = RttmUtils(OUT_DIR)
    rttm_utils.check_the_results_of_diarization(file_name=filename)

    diar_segments = rttm_utils.format_diarization_rttm()

    for diar_segment in diar_segments:
        logger.info(diar_segments)

    # Merging diar_segments with whisper_segments
    # Each Whisper segment (wseg) can overlap with several diar segments.

    # methods = [DiarizationService.probabilistic_speaker_matching]

    word_level_segments = DiarizationService.word_level_alignment(
        whisper_segments,  # c полем "words"
        diar_segments      # [{start, end, speaker}, ...]
    )

    # 2) Склеиваем в более крупные фразы
    final_merged = DiarizationService.merge_word_level_results(word_level_segments)

    # 3) (Опционально) Можно ещё раз прогнать ваш merge_adjacent_segments, если нужно
    #    или оставить merge_word_level_results как основную "склейку".
    # final_merged = DiarizationService.merge_adjacent_segments(final_merged)


    ##########################################################################
    # D) Final merge
    ##########################################################################

    DiarizationService.export_results(final_merged, None)
    logger.info("\n=== FINAL MERGED (WHISPER + MSDD DIARIZATION) ===")


if __name__ == "__main__":
    main()

