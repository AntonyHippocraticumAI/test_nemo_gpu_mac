import json
import os
import librosa
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf
import torch

from services.model_manager.vad_manager import preload_vad_model
from services.model_manager.whisper_manager import preload_models
from utils.d_utils import DiarizationService
from utils.manifest import create_manifest
from utils.config_models import create_config_diarisation
from utils.rttm_utils import RttmUtils
from utils.settings import AUDIO_PATH, VAD_MODEL_PATH, EMBED_MODEL_PATH
from utils.logging_utils import get_logger



def create_config_diarisation(MANIFEST_PATH, OUT_DIR):
    diar_config = {
        "device": "cuda",
        "num_workers": 1,
        "batch_size": 64,
        "sample_rate": 16000,
        "verbose": True,

        "diarizer": {
            "manifest_filepath": MANIFEST_PATH,
            "out_dir": OUT_DIR,
            "oracle_vad": False,  # використ. системний VAD
            "collar": 0.25,
            "ignore_overlap": True,

            "vad": {
                "model_path": VAD_MODEL_PATH,
                'external_vad_manifest': None,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.875,
                    "onset": 0.5,  # Увеличен порог для уверенного обнаружения начала речи
                    "offset": 0.5,  # Сбалансированный порог для конца речи
                    "pad_onset": 0.1,  # Увеличен для захвата начала фраз
                    "pad_offset": 0.1,  # Увеличен для захвата концов фраз
                    "min_duration_on": 0.25,  # Увеличен для игнорирования коротких шумов
                    "min_duration_off": 0.15,  # Уменьшен для лучшего разделения фраз
                    "filter_speech_first": True
                }
            },
            "speaker_embeddings": {
                "model_path": EMBED_MODEL_PATH,
                "infer_batches": True,
                "parameters": {
                    "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5], # Window length(s) in sec (floating-point number). Either a number or a list. Ex) 1.5 or [1.5,1.25,1.0,0.75,0.5]
                    "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25], # Shift length(s) in sec (floating-point number). Either a number or a list. Ex) 0.75 or [0.75,0.625,0.5,0.375,0.25]
                    "multiscale_weights": [1, 1, 1, 1, 1], # Weight for each scale. should be null (for single scale) or a list matched with window/shift scale count. Ex) [1,1,1,1,1]
                    "save_embeddings": False # Save embeddings as pickle file for each audio input.
                }
            },
            "clustering": {
                # Якщо точно 2 спікери:
                # "kmeans_num_clusters": 2,
                # "oracle_num_speakers": False,

                # Якщо auto:
                "kmeans_num_clusters": None,
                "oracle_num_speakers": False,
                "parameters": {
                    "oracle_num_speakers": False,
                    "max_num_speakers": 8,  # Уменьшено для диалогов
                    "enhanced_count_thres": 40,  # Снижено для лучшей работы с короткими аудио
                    "max_rp_threshold": 0.15,  # Снижено для более строгой кластеризации
                    "sparse_search_volume": 40,  # Увеличено для более точного поиска
                    "maj_vote_spk_count": True,  # Включен режим мажоритарного голосования
                    "chunk_cluster_count": 30,  # Оптимизировано
                    "embeddings_per_chunk": 5000  # Уменьшено для коротких диалогов
                }
            },
            "msdd_model": {
                # Параметри MSDD. Якщо треба можна вказати "model_path": "diar_msdd_telephonic..."
                # або лишити порожнім, тоді NeMo завантажить дефолтну модель.
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_speaker_model_from_ckpt": True,
                    "infer_batch_size": 25,
                    "sigmoid_threshold": [0.7, 0.6, 0.5],  # Несколько порогов для повышения точности
                    "seq_eval_mode": False,
                    "split_infer": True,
                    "diar_window_length": 30,  # Уменьшено для лучшей работы с короткими аудио
                    "overlap_infer_spk_limit": 3,  # Подходит для диалогов
                }
            },
            "enable_msdd": True,  # вмикаємо MSDD
            "convert_to_unique_speaker_ids": True,  # видаватиме speaker_0, speaker_1, ...
        }
    }

    cfg = OmegaConf.create(diar_config)

    return cfg

def create_manifest(audio_path, manifest_path, num_speakers=None):
    """
    Создает улучшенный манифест для диаризации
    
    Parameters:
    audio_path (str): Путь к аудиофайлу
    manifest_path (str): Путь для сохранения манифеста
    num_speakers (int, optional): Количество спикеров (если известно)
    """
    import json
    import os
    from pathlib import Path
    
    audio_path = os.path.abspath(audio_path)
    
    # Получаем длительность аудиофайла
    import librosa
    duration = librosa.get_duration(path=audio_path)
    
    manifest = {
        "audio_filepath": audio_path,
        "offset": 0,
        "duration": duration,
        "label": "infer",
        "rttm_filepath": None,
     }
    
    # Добавляем информацию о количестве спикеров, если она известна
    if num_speakers is not None and num_speakers > 0:
        manifest["num_speakers"] = int(num_speakers)
    
    # Сохраняем манифест
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    
    return manifest_path

def enhance_merge_transcription_with_speakers(whisper_segments, diar_segments, min_overlap_ratio=0.2):
    """
    Улучшенное объединение сегментов Whisper с диаризацией с дополнительными проверками
    
    Parameters:
    whisper_segments (List[Dict]): Сегменты Whisper с 'start', 'end', 'text'
    diar_segments (List[Dict]): Сегменты диаризации с 'speaker', 'start', 'end'
    min_overlap_ratio (float): Минимальное соотношение перекрытия для уверенного определения
    
    Returns:
    List[Dict]: Объединенные сегменты с назначенными спикерами
    """
    result = []
    
    # Подготовка сегментов диаризации для более эффективного поиска
    diar_segments = sorted(diar_segments, key=lambda x: x['start'])
    
    for transcript in whisper_segments:
        t_start = transcript['start']
        t_end = transcript['end']
        t_text = transcript['text']
        t_duration = t_end - t_start
        
        # Найдем все перекрывающиеся сегменты диаризации
        overlapping_segments = []
        
        for diar in diar_segments:
            d_start = diar['start']
            d_end = diar['end']
            
            # Проверка на перекрытие
            if d_end <= t_start or d_start >= t_end:
                continue
            
            # Вычисление перекрытия
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap_duration = overlap_end - overlap_start
            
            # Добавляем сегмент и его перекрытие
            overlapping_segments.append({
                'speaker': diar['speaker'],
                'overlap_duration': overlap_duration,
                'overlap_ratio': overlap_duration / t_duration
            })
        
        # Если нет перекрывающихся сегментов
        if not overlapping_segments:
            result.append({
                'start': t_start,
                'end': t_end,
                'text': t_text,
                'speaker': "unknown",
                'confidence': 0.0
            })
            continue
        
        # Сортировка по продолжительности перекрытия
        overlapping_segments.sort(key=lambda x: x['overlap_ratio'], reverse=True)
        
        # Проверка на уверенное определение спикера
        best_overlap = overlapping_segments[0]
        confidence = best_overlap['overlap_ratio']
        
        if confidence >= min_overlap_ratio:
            speaker = best_overlap['speaker']
        else:
            # Подсчет общего времени по каждому спикеру
            speaker_times = {}
            for seg in overlapping_segments:
                spk = seg['speaker']
                if spk not in speaker_times:
                    speaker_times[spk] = 0
                speaker_times[spk] += seg['overlap_duration']
            
            # Выбор спикера с наибольшим общим временем
            speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
            confidence = best_overlap['overlap_ratio']  # Используем как меру уверенности
        
        result.append({
            'start': t_start,
            'end': t_end,
            'text': t_text,
            'speaker': speaker,
            'confidence': confidence
        })
    
    # Пост-обработка для исправления непоследовательностей
    return post_process_speaker_assignments(result)

def post_process_speaker_assignments(merged_segments, context_window=3):
    """
    Пост-обработка для улучшения согласованности назначения спикеров
    
    Parameters:
    merged_segments (List[Dict]): Объединенные сегменты
    context_window (int): Размер окна контекста для анализа
    
    Returns:
    List[Dict]: Обработанные сегменты
    """
    processed = merged_segments.copy()
    
    # Проход 1: Исправление изолированных сегментов
    for i in range(1, len(processed) - 1):
        if processed[i-1]['speaker'] == processed[i+1]['speaker'] and processed[i]['speaker'] != processed[i-1]['speaker']:
            # Текущий сегмент отличается от предыдущего и следующего
            if processed[i]['confidence'] < 0.6:  # Порог уверенности для коррекции
                processed[i]['speaker'] = processed[i-1]['speaker']
                processed[i]['confidence'] = 0.5  # Пониженная уверенность для исправленных сегментов
    
    # Проход 2: Анализ диалоговой структуры (например, вопрос-ответ)
    for i in range(len(processed) - 1):
        current = processed[i]['text'].strip()
        next_text = processed[i+1]['text'].strip()
        
        # Проверка на вопрос-ответ паттерны
        if current.endswith('?') and not next_text.endswith('?') and processed[i]['speaker'] == processed[i+1]['speaker']:
            # Вероятно, это вопрос и ответ - должны быть разные спикеры
            if i+2 < len(processed) and processed[i+2]['speaker'] != processed[i+1]['speaker']:
                # Используем спикера из сегмента после следующего
                processed[i+1]['speaker'] = processed[i+2]['speaker']
                processed[i+1]['confidence'] = 0.5
    
    return processed

def format_enhanced_transcript(merged_segments, use_confidence=False, rename_speakers=True):
    """
    Улучшенное форматирование транскрипции с опциями для удобочитаемости
    
    Parameters:
    merged_segments (List[Dict]): Объединенные сегменты
    use_confidence (bool): Включить информацию об уверенности
    rename_speakers (bool): Переименовать speaker_X в более читаемые имена
    
    Returns:
    str: Отформатированная транскрипция
    """
    formatted_transcript = []
    current_speaker = None
    current_text = []
    
    # Определение ролей говорящих на основе контекста (для переименования)
    speaker_roles = {}
    
    if rename_speakers:
        # Анализируем первые несколько сегментов для определения ролей
        for segment in merged_segments[:5]:
            text = segment['text'].lower()
            speaker = segment['speaker']
            
            if "dr." in text or "doctor" in text or "check" in text or "test" in text or "exam" in text:
                speaker_roles[speaker] = "Doctor"
            elif "pain" in text or "hurts" in text or "tried" in text:
                speaker_roles[speaker] = "Patient"
        
        # Если не удалось определить, используем дефолтные имена
        if len(speaker_roles) < 2:
            speakers = list(set(s['speaker'] for s in merged_segments if s['speaker'] != 'unknown'))
            if len(speakers) >= 1 and speakers[0] not in speaker_roles:
                speaker_roles[speakers[0]] = "Speaker A"
            if len(speakers) >= 2 and speakers[1] not in speaker_roles:
                speaker_roles[speakers[1]] = "Speaker B"
    
    for segment in merged_segments:
        speaker = segment['speaker']
        text = segment['text']
        confidence = segment.get('confidence', 1.0)
        
        # Применяем переименование спикеров если нужно
        display_speaker = speaker
        if rename_speakers and speaker in speaker_roles:
            display_speaker = speaker_roles[speaker]
        
        if speaker != current_speaker:
            # Если сменился говорящий, форматируем предыдущий текст
            if current_speaker and current_text:
                prev_display_speaker = current_speaker
                if rename_speakers and current_speaker in speaker_roles:
                    prev_display_speaker = speaker_roles[current_speaker]
                
                formatted_transcript.append(f"{prev_display_speaker}: {' '.join(current_text)}")
                current_text = []
            
            current_speaker = speaker
        
        # Добавляем текущий текст
        if use_confidence and confidence < 0.7:
            current_text.append(f"{text} [conf:{confidence:.2f}]")
        else:
            current_text.append(text)
    
    # Добавляем последний блок текста
    if current_speaker and current_text:
        display_speaker = current_speaker
        if rename_speakers and current_speaker in speaker_roles:
            display_speaker = speaker_roles[current_speaker]
        
        formatted_transcript.append(f"{display_speaker}: {' '.join(current_text)}")
    
    return "\n\n".join(formatted_transcript)


def preprocess_audio_for_diarization(audio_path, output_path=None):
    """
    Предобработка аудио для улучшения качества диаризации
    
    Parameters:
    audio_path (str): Путь к входному аудиофайлу
    output_path (str, optional): Путь для сохранения обработанного аудио
    
    Returns:
    str: Путь к обработанному аудиофайлу
    """
    import librosa
    import soundfile as sf
    import numpy as np
    from scipy import signal
    
    if output_path is None:
        output_path = "enhanced_audio_for_diarization.wav"
    
    # Загрузка аудио
    y, sr = librosa.load(audio_path, sr=16000)
    
    # 1. Нормализация громкости
    y = librosa.util.normalize(y)
    
    # 2. Удаление шума (простой фильтр высоких частот)
    b, a = signal.butter(4, 100/(sr/2), 'highpass')
    y = signal.filtfilt(b, a, y)
    
    # 3. Компрессия динамического диапазона для улучшения речи
    def compress_dynamic_range(audio, threshold=0.05, ratio=2.0):
        indices = np.where(np.abs(audio) > threshold)[0]
        audio[indices] = np.sign(audio[indices]) * (
            threshold + (np.abs(audio[indices]) - threshold) / ratio
        )
        return audio
    
    y = compress_dynamic_range(y)
    
    # 4. Повторная нормализация
    y = librosa.util.normalize(y)
    
    # Сохранение обработанного аудио
    sf.write(output_path, y, sr)
    
    return output_path

def main(audio_path, known_speakers=0):
    """
    Основная функция обработки с улучшенными параметрами
    
    Parameters:
    audio_path (str): Путь к аудиофайлу
    known_speakers (int): Количество спикеров (если известно)
    
    Returns:
    tuple: (merged_segments, formatted_transcript)
    """
    import os
    import json
    
    # Создаем директории
    ROOT_DIR = "./nemo_diar_temp"
    os.makedirs(ROOT_DIR, exist_ok=True)
    MANIFEST_PATH = os.path.join(ROOT_DIR, "input_manifest.json")
    OUT_DIR = os.path.join(ROOT_DIR, "diar_output")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Предобработка аудио
    enhanced_audio = preprocess_audio_for_diarization(audio_path)
    
    # 2. Загрузка моделей
    whisper_model_manager = preload_models()
    vad_model_manager = preload_vad_model()
    
    # 3. Применение VAD для улучшения сегментации
    audio_without_vad, sr = vad_model_manager.get_audio_time_series(enhanced_audio)
    params_vad = vad_model_manager.get_params_for_vad_model(
        audio_without_vad, sr, 
        window_length=0.2,  # Меньше для лучшего разделения
        onset_thresh=0.5,   # Более высокий порог для четкого начала
        min_speech_dur=0.2  # Короче для захвата кратких высказываний
    )
    
    segments = vad_model_manager.get_speech_segments(**params_vad)
    audio_data = vad_model_manager.get_ndarray_of_segments(audio_without_vad, segments, sr)
    
    # Сохранение очищенного аудио
    import numpy as np
    from scipy.io.wavfile import write
    
    if audio_data.dtype != np.int16:
        audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data = np.int16(audio_data * 32767)
    
    cleaned_audio_path = os.path.join(ROOT_DIR, "cleaned_audio.wav")
    write(cleaned_audio_path, 16000, audio_data)
    
    # 4. Транскрипция Whisper с оптимизированными параметрами
    import librosa
    audio, sr = librosa.load(cleaned_audio_path, sr=16000)
    
    segments_gen, _ = whisper_model_manager.transcribe_for_diarisation(
        audio, 
        model_name="large-v3"
    )
    whisper_segments = whisper_model_manager.produced_segments(segments_gen)
    
    # 5. Диаризация с улучшенной конфигурацией
    create_manifest(cleaned_audio_path, MANIFEST_PATH)
    cfg = create_config_diarisation(MANIFEST_PATH, OUT_DIR)
    
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    diar_model = NeuralDiarizer(cfg=cfg)

    diar_model.diarize()


    # 6. Чтение результатов RTTM
    rttm_utils = RttmUtils(OUT_DIR)
    rttm_utils.check_the_results_of_diarization()
    diar_segments = rttm_utils.format_diarization_rttm()
    
    # 7. Улучшенное объединение транскрипции и диаризации
    merged_segments = enhance_merge_transcription_with_speakers(whisper_segments, diar_segments)
    formatted_transcript = format_enhanced_transcript(merged_segments, rename_speakers=True)
    
    # 8. Сохранение результатов
    with open(os.path.join(OUT_DIR, "transcript_with_speakers.json"), "w", encoding="utf-8") as f:
        json.dump(merged_segments, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUT_DIR, "transcript_with_speakers.txt"), "w", encoding="utf-8") as f:
        f.write(formatted_transcript)
    
    return merged_segments, formatted_transcript


if __name__ == "__main__":
    
    main(AUDIO_PATH)