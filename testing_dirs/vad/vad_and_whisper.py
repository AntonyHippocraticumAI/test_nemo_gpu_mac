import os
from datetime import datetime, time
from time import time as test_time

import torch
import librosa
import numpy as np
import soundfile as sf

from nemo.collections.asr.models import EncDecClassificationModel
from services.model_manager.whisper_manager import preload_models


def get_speech_segments(audio, sr, vad_model,
                        window_length=0.63,
                        shift_length=0.08,
                        onset_thresh=0.5,
                        offset_thresh=0.3,
                        pad_onset=0.2,
                        pad_offset=0.2,
                        min_speech_dur=0.5):
    """
    Розбиває аудіо на фрейми, проганяє через VAD-модель та
    повертає список часових сегментів, де є мовлення.
    """

    # Перетворюємо секунди у кількість семплів
    window_samples = int(window_length * sr)
    shift_samples = int(shift_length * sr)

    speech_probs = []  # зберігатимемо (початок_у_семплах, ймовірність_мови)
    start_idx = 0

    while start_idx + window_samples <= len(audio):
        chunk = audio[start_idx : start_idx + window_samples]

        # перетворюємо на тензор
        chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        if torch.cuda.is_available():
            chunk_tensor = chunk_tensor.cuda()

        # Отримуємо логіти від VAD-моделі
        with torch.no_grad():
            logits = vad_model.forward(
                input_signal=chunk_tensor,
                input_signal_length=torch.tensor([chunk_tensor.shape[1]], dtype=torch.long)
            )
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # для VAD: індекс 0 = non_speech, індекс 1 = speech
        speech_prob = probs[1]
        speech_probs.append((start_idx, speech_prob))

        start_idx += shift_samples

    # Подальша обробка, щоб отримати (start_time, end_time)
    segments = []
    is_speech = False
    seg_start = 0.0

    for (idx, prob) in speech_probs:
        t_sec = idx / sr
        # Якщо зараз "тишина" і починається мовлення
        if not is_speech and prob >= onset_thresh:
            is_speech = True
            seg_start = max(0, t_sec - pad_onset)

        # Якщо зараз "мовлення" і ймовірність падає нижче offset
        elif is_speech and prob < offset_thresh:
            is_speech = False
            seg_end = min(len(audio)/sr, t_sec + pad_offset)
            if seg_end - seg_start >= min_speech_dur:
                segments.append((seg_start, seg_end))

    # Якщо аудіо закінчилось, а ми досі "в мовленні"
    if is_speech:
        seg_end = len(audio)/sr
        if seg_end - seg_start >= min_speech_dur:
            segments.append((seg_start, seg_end))

    return segments


def generate_readable_filename(original_filename: str, batch_flag: str) -> str:
    base_name, ext = os.path.splitext(f"/Users/antonandreev/python_prog/test_nemo_cpu_mac/results/vad_and_transcriptions/{batch_flag}{original_filename}")
    timestamp = datetime.now().strftime("%Y.%m.%d---%H:%M:%S")
    return f"{base_name}_{timestamp}{ext}"


def split_by_words(text, words_per_line=10):
    words = text.split()
    lines = [" ".join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
    return "\n".join(lines)


def main(name_model: str, vad_model_path:str, mode_whisper: str = "", batch_flag: bool = False):
    # === 1) Завантажимо VAD-модель ===
    vad_model = EncDecClassificationModel.restore_from(vad_model_path)
    vad_model.eval()
    if torch.cuda.is_available():
        vad_model = vad_model.cuda()

    # === 2) Зчитуємо аудіо ===
    input_audio_path = "/Users/antonandreev/python_prog/test_nemo_cpu_mac/audio_samples/recording_16000hz.wav"
    audio, sr = librosa.load(input_audio_path, sr=16000)  # Ставимо 16 кГц


    start_vad = test_time()
    # === 3) Знаходимо сегменти мовлення ===
    speech_segments = get_speech_segments(
        audio, sr, vad_model,
        window_length=0.63,
        shift_length=0.08,
        onset_thresh=0.5,
        offset_thresh=0.3,
        pad_onset=0.2,
        pad_offset=0.2,
        min_speech_dur=0.5
    )

    print("Знайдено сегменти мовлення:", speech_segments)

    # === 4) Вирізаємо тишину й зберігаємо у новий файл ===
    cleaned_audio = []
    for (start_sec, end_sec) in speech_segments:
        start_i = int(start_sec * sr)
        end_i = int(end_sec * sr)
        cleaned_audio.append(audio[start_i:end_i])

    if len(cleaned_audio) > 0:
        cleaned_audio = np.concatenate(cleaned_audio)
    else:
        # Якщо раптом немає мовлення
        cleaned_audio = np.array([])

    output_audio_path = "cleaned_audio.wav"
    sf.write(output_audio_path, cleaned_audio, sr)

    print(f"CREATE FILE WITHOUT SILENCE: {output_audio_path}")
    end_vad = round(test_time() - start_vad, 2)
    print("VAD WORK:", end_vad)

    # === 5) Транскрибуємо "очищений" файл за допомогою faster-whisper ===
    whisper_model_manager = preload_models()
    recognized_text, info = whisper_model_manager.transcribe_with_vad(audio_waveform=output_audio_path,
                                                                      batch_flag=batch_flag)

    print("Результат розпізнавання (Whisper):")
    print(recognized_text, info)

    try:
        filename_vad = generate_readable_filename(name_model, mode_whisper)

        output_dir = os.path.dirname(filename_vad)

        os.makedirs(output_dir, exist_ok=True)

        formatted_text = split_by_words(recognized_text, words_per_line=10)

        with open(f"{os.path.splitext(filename_vad)[0]}.txt", "w", encoding="utf-8-sig") as f:
            f.write("--------------------------------------------------------\n\n")

            if mode_whisper:
                batch_mode = "TURN ON"
            else:
                batch_mode = "TURN OFF"

            f.write(f"BATCH MODE: {batch_mode}; VAD WORK TIME - {end_vad}...TESTED ON MAC1 PRO\n\n")

            f.write("VAD SEGMENTS:\n")
            for vad_segment in enumerate(speech_segments):
                f.write(f"\t{vad_segment[0]}: {vad_segment[1]}\n")

            f.write("\n")

            if mode_whisper:
                f.write("TRANSCRIPTIONS:\n")
                f.write(f"\t{formatted_text}\n\n")
            else:
                f.write("TRANSCRIPTIONS:\n")
                f.write(f"\t{formatted_text}\n\n")

                for segment in enumerate(info):
                    f.write(f"\t{segment[0]}: {segment[1]}\n")

    except Exception as e:
        raise e


if __name__ == "__main__":

    name_model = "commandrecognition_en_matchboxnet3x2x64_v2_subset_task.nemo"
    vad_model_path = "/Users/antonandreev/commandrecognition_en_matchboxnet3x2x64_v2_subset_task_v1.0.0rc1/commandrecognition_en_matchboxnet3x2x64_v2_subset_task.nemo"
    mode_whisper = "with_batch_" # with_batch_ or empty string
    batch_flag = True

    main(name_model, vad_model_path, mode_whisper, batch_flag)
