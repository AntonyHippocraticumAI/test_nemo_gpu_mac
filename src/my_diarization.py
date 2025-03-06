import os
import librosa
from nemo.collections.asr.models import NeuralDiarizer

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
    # A) TRANSCRIPTION WHISPER
    ##########################################################################
    audio, sr = librosa.load(AUDIO_PATH, sr=16000)
    logger.info(f"Loaded audio {AUDIO_PATH}, sr={sr}, duration={len(audio)/sr:.1f}s")

    logger.info("=== WHISPER STEP ===")
    whisper_model_manager = preload_models()

    segments_gen, _ = whisper_model_manager.transcribe_for_diarisation(audio)
    logger.info(segments_gen)
    whisper_segments = whisper_model_manager.produced_segments(segments_gen)

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
    rttm_utils.check_the_results_of_diarization()

    diar_segments = rttm_utils.format_diarization_rttm()

    # Merging diar_segments with whisper_segments
    # Each Whisper segment (wseg) can overlap with several diar segments.

    # methods = [DiarizationService.probabilistic_speaker_matching]

    matched_segments = DiarizationService.weighted_ensemble_speaker_matching(
        whisper_segments,
        diar_segments,
        # methods=methods,
    )

    final_merged = DiarizationService.improved_merge_adjacent_segments(matched_segments)

    ##########################################################################
    # D) Final merge
    ##########################################################################

    DiarizationService.export_results(final_merged, None)
    logger.info("\n=== FINAL MERGED (WHISPER + MSDD DIARIZATION) ===")


if __name__ == "__main__":
    main()
