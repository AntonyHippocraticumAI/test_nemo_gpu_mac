import threading
from functools import lru_cache

import librosa
import torch
import numpy as np
from nemo.collections.asr.models import EncDecClassificationModel

from utils.settings import VAD_MODEL_PATH, device


class NemoVADModelManager:
    def __init__(self):
        """
        :param vad_model_path: шлях до VAD .nemo-файлу
        :param device: "cpu" або "cuda" (якщо є GPU)
        """
        self.vad_model_path = VAD_MODEL_PATH
        self.device = device

        self._lock = threading.Lock()
        self._loaded_vad_model = None  # Модель завантажимо ледачо

    def load_vad_model(self) -> EncDecClassificationModel:
        """Потокобезпечне завантаження VAD-моделі з файлу."""
        with self._lock:
            if self._loaded_vad_model is None:
                print(f"Loading VAD model from: {self.vad_model_path}")
                model = EncDecClassificationModel.restore_from(self.vad_model_path)
                model.eval()

                if self.device == "cuda" and torch.cuda.is_available():
                    model = model.cuda()

                self._loaded_vad_model = model
                print("VAD model loaded successfully.")

            return self._loaded_vad_model


    def get_vad_model(self) -> EncDecClassificationModel:
        """
        Зручний метод, щоб просто отримати (вже завантажену) VAD-модель.
        Якщо модель ще не завантажена, буде викликано load_vad_model().
        """
        return self.load_vad_model()


    def get_audio_time_series(self, input_audio_path: str, sr=16000):
        return librosa.load(input_audio_path, sr=sr)


    def get_params_for_vad_model(self,
                                 audio_waveform: np.ndarray,
                                 sr: int = 16000,
                                 window_length: float = 0.63,
                                 shift_length: float = 0.08,
                                 onset_thresh: float = 0.5,
                                 offset_thresh: float = 0.3,
                                 pad_onset: float = 0.2,
                                 pad_offset: float = 0.2,
                                 min_speech_dur: float = 0.5
    ):
        return {
            "audio_waveform": audio_waveform,
            "sr": sr,
            "window_length": window_length,
            "shift_length": shift_length,
            "onset_thresh": onset_thresh,
            "offset_thresh": offset_thresh,
            "pad_onset": pad_onset,
            "pad_offset": pad_offset,
            "min_speech_dur": min_speech_dur,
        }


    def get_speech_segments(self,
                            audio_waveform: np.ndarray,
                            sr: int = 16000,
                            window_length: float = 0.63,
                            shift_length: float = 0.08,
                            onset_thresh: float = 0.5,
                            offset_thresh: float = 0.3,
                            pad_onset: float = 0.2,
                            pad_offset: float = 0.2,
                            min_speech_dur: float = 0.5) -> list[tuple[float, float]]:
        """
        Приклад методу, який робить інференс VAD-моделі і повертає сегменти (start_sec, end_sec).
        Код можна адаптувати під ваші потреби.

        :param audio_waveform: np.ndarray зі звуковими даними
        :param sr: частота дискретизації (за замовчуванням 16 кГц)
        :param window_length, shift_length, onset_thresh, offset_thresh: гіперпараметри VAD
        :param pad_onset, pad_offset: "запас" по часу перед/після мовного фрагмента
        :param min_speech_dur: мінімальна тривалість мовного фрагмента, що не відсікається
        :return: список кортежів [(start_time, end_time), ...]
        """
        model = self.get_vad_model()  # Гарантовано отримаємо VAD-модель

        # Ваша логіка для VAD (розбити сигнал на вікна, отримати logits, зібрати сегменти).
        # Для прикладу - спрощена версія (без згладжування):
        window_samples = int(window_length * sr)
        shift_samples = int(shift_length * sr)

        speech_probs = []
        start_i = 0

        while start_i + window_samples <= len(audio_waveform):
            chunk = audio_waveform[start_i: start_i + window_samples]

            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            if self.device == "cuda" and torch.cuda.is_available():
                chunk_tensor = chunk_tensor.cuda()

            with torch.no_grad():
                logits = model.forward(
                    input_signal=chunk_tensor,
                    input_signal_length=torch.tensor([chunk_tensor.shape[1]], dtype=torch.long)
                )
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            speech_prob = probs[1]  # індекс 1 = speech

            speech_probs.append((start_i, speech_prob))

            start_i += shift_samples

        # перетворюємо speech_probs -> [(start_sec, end_sec), ...]
        segments = []
        is_speech = False
        seg_start = 0.0

        for (idx, prob) in speech_probs:
            t_sec = idx / sr

            if not is_speech and prob >= onset_thresh:
                is_speech = True
                seg_start = max(0, t_sec - pad_onset)
            elif is_speech and prob < offset_thresh:
                is_speech = False
                seg_end = min(len(audio_waveform) / sr, t_sec + pad_offset)
                if seg_end - seg_start >= min_speech_dur:
                    segments.append((seg_start, seg_end))

        # Якщо аудіо закінчилося, а ми досі "в мовленні"
        if is_speech:
            seg_end = len(audio_waveform) / sr
            if seg_end - seg_start >= min_speech_dur:
                segments.append((seg_start, seg_end))

        return segments


    def get_ndarray_of_segments(self, audio, segments: list[tuple[float, float]], sr: int = 16000) -> np.ndarray:
        chunks = []
        for (start_sec, end_sec) in segments:
            start_i = int(start_sec * sr)
            end_i = int(end_sec * sr)
            chunks.append(audio[start_i:end_i])


        if chunks:
            cleaned_audio = np.concatenate(chunks)
        else:
            cleaned_audio = np.array([])

        return cleaned_audio


@lru_cache
def get_vad_manager() -> NemoVADModelManager:
    """
    Повертає екземпляр NemoVADModelManager.
    Завдяки @lru_cache для кожної унікальної конфігурації
    буде створено лише один екземпляр на весь додаток.
    """
    return NemoVADModelManager()


def preload_vad_model():
    """
    Опційна функція, щоб випереджувально (preload) завантажити модель.
    Викликати, наприклад, під час ініціалізації сервісу.
    """
    manager = get_vad_manager()
    print("Preloading VAD model ...")
    manager.load_vad_model()
    print("VAD model is ready!")
    return manager