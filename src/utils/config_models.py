from omegaconf import OmegaConf

from .settings import VAD_MODEL_PATH, EMBED_MODEL_PATH


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
            "collar": 0.15,
            "ignore_overlap": False,

            "vad": {
                "model_path": VAD_MODEL_PATH,
                'external_vad_manifest': None,
                "parameters": {
                    "window_length_in_sec": 0.15,  # Window length in sec for VAD context input
                    "shift_length_in_sec": 0.01,  # Shift length in sec for generate frame level VAD prediction
                    "smoothing": "median",  # False or type of smoothing method (eg: median)
                    "overlap": 0.875,  # Overlap ratio for overlapped mean/median smoothing filter
                    "onset": 0.5,  # Onset threshold for detecting the beginning and end of a speech
                    "offset": 0.5,  # Offset threshold for detecting the end of a speech
                    "pad_onset": 0.1,  # Adding durations before each speech segment
                    "pad_offset": 0.1,  # Adding durations after each speech segment
                    "min_duration_on": 0.25,  # Threshold for small non_speech deletion
                    "min_duration_off": 0.15,  # Threshold for short speech segment deletion
                    "filter_speech_first": True
                }
            },
            "speaker_embeddings": {
                "model_path": EMBED_MODEL_PATH,
                "infer_batches": True,
                "parameters": {
                    "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                    # Window length(s) in sec (floating-point number). either a number or a list. ex) 1.5 or [1.5,1.0,0.5]
                    "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                    "multiscale_weights": [1, 1, 1, 1, 1],  # Равномерные веса
                    "save_embeddings": True
                }
            },
            "clustering": {
                # Если известно точное количество говорящих (например, 2 для диалога врач-пациент):
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
                    # If True, use speaker embedding model in checkpoint. If False, the provided speaker embedding model in config will be used.
                    "infer_batch_size": 25,  # Batch size for MSDD inference.
                    "sigmoid_threshold": [0.7, 0.6, 0.5],
                    # Sigmoid threshold for generating binarized speaker labels. The smaller the more generous on detecting overlaps.
                    "seq_eval_mode": False,
                    # If True, use oracle number of speaker and evaluate F1 score for the given speaker sequences. Default is False.
                    "split_infer": True,
                    # If True, break the input audio clip to short sequences and calculate cluster average embeddings for inference.
                    "diar_window_length": 30,  # The length of split short sequence when split_infer is True.
                    "overlap_infer_spk_limit": 3,
                    # If the estimated number of speakers are larger than this number, overlap speech is not estimated.

                }
            },
            "enable_msdd": True,  # вмикаємо MSDD
            "convert_to_unique_speaker_ids": True,  # видаватиме speaker_0, speaker_1, ...
        }
    }

    cfg = OmegaConf.create(diar_config)

    return cfg