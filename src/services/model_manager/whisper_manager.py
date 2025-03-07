import threading
from functools import lru_cache

import faster_whisper
from faster_whisper import WhisperModel

from utils.settings import suppress_numerals, whisper_model_name, device, compute_type, batch_size, beams_size
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class WhisperModelManager:
    def __init__(self):
        self.model = whisper_model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.beams_size = beams_size

        self.loaded_models = {}
        self._lock = threading.Lock()

    def load_model(self, model_name: str) -> WhisperModel:
        with self._lock:
            if model_name not in self.loaded_models:
                self.loaded_models[model_name] = WhisperModel(
                    model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    num_workers=10,
                )
            return self.loaded_models[model_name]


    def get_model(self, model_name: str) -> WhisperModel:
        return self.load_model(model_name)


    def transcribe_with_vad(self,
                            audio_waveform,
                            model_name: str = whisper_model_name,
                            batch_flag: bool = False,
    ):
        whisper_model = self.get_model(model_name)

        if batch_flag:

            whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

            try:
                transcript_segments, info = whisper_pipeline.transcribe(
                    audio_waveform,
                    batch_size=self.batch_size,
                    beam_size=self.beams_size,
                )

                full_transcript = "".join(segment.text for segment in transcript_segments)

                return full_transcript, info
            except Exception as e:
                raise e

        try:
            transcript_segments, info = whisper_model.transcribe(audio_waveform, beam_size=5)

            final_text = []
            for segment in transcript_segments:
                final_text.append(segment.text)
                logger.info(segment.text)
        except Exception as e:
            logger.error(e)
            raise e

        full_transcript = "".join(segment for segment in final_text)

        return full_transcript, final_text


    def transcribe_for_diarisation(self, audio_waveform, model_name: str = whisper_model_name):
        whisper_model = self.get_model(model_name)

        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

        try:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                beam_size=5,
                best_of=5,
                temperature=0,  # Более точное распознавание
                compression_ratio_threshold=2.4,  # Регулировка для более точного разделения предложений
                condition_on_previous_text=True,  # Учитывать предыдущий текст
                no_speech_threshold=0.6,  # Увеличен порог для четкого разделения речи
                without_timestamps=False,  # Важно для диаризации
                word_timestamps=True,  # Можно включить для более детальной выверки
                patience=1,  # Параметр для beam search
                max_initial_timestamp=0.5,  # Максимальное смещение первого сегмента
                suppress_blank=True,  # Подавлять пустые выводы
            )
        except Exception as e:
            raise e

        return transcript_segments, info


    def produced_segments(self, segments_generation):
        whisper_segments = []
        for seg in segments_generation:
            whisper_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip().replace("\n", " "),
            })
        return whisper_segments


@lru_cache
def get_model_manager() -> WhisperModelManager:
    return WhisperModelManager()


def preload_models():
    manager = get_model_manager()
    model_name = whisper_model_name

    logger.info(f"Loading model: {model_name}")
    try:
        manager.load_model(model_name)
        logger.info(f"Model {model_name} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        logger.error(e)
        raise

    return manager
