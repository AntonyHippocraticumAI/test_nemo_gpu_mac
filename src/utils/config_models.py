from omegaconf import OmegaConf

from utils.settings import VAD_MODEL_PATH, EMBED_MODEL_PATH


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
                    "window_length_in_sec": 0.15,  # Window length in sec for VAD context input
                    "shift_length_in_sec": 0.01,  # Shift length in sec for generate frame level VAD prediction
                    "smoothing": "median",  # False or type of smoothing method (eg: median)
                    "overlap": 0.875,  # Overlap ratio for overlapped mean/median smoothing filter
                    "onset": 0.4,  # Onset threshold for detecting the beginning and end of a speech
                    "offset": 0.7,  # Offset threshold for detecting the end of a speech
                    "pad_onset": 0.05,  # Adding durations before each speech segment
                    "pad_offset": -0.1,  # Adding durations after each speech segment
                    "min_duration_on": 0.2,  # Threshold for small non_speech deletion
                    "min_duration_off": 0.2,  # Threshold for short speech segment deletion
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
                    # Shift length(s) in sec (floating-point number). either a number or a list. ex) 0.75 or [0.75,0.5,0.25]
                    "multiscale_weights": [1, 1, 1, 1, 1],
                    # Weight for each scale. should be null (for single scale) or a list matched with window/shift scale count. ex) [0.33,0.33,0.33]
                    "save_embeddings": True
                    # If True, save speaker embeddings in pickle format. This should be True if clustering result is used for other models, such as `msdd_model`.

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
                    "oracle_num_speakers": False,  # If True, use num of speakers value provided in manifest file.
                    "max_num_speakers": 20,
                    # Max number of speakers for each recording. If an oracle number of speakers is passed, this value is ignored.
                    "enhanced_count_thres": 80,
                    # If the number of segments is lower than this number, enhanced speaker counting is activated.
                    "max_rp_threshold": 0.25,  # Determines the range of p-value search: 0 < p <= max_rp_threshold.
                    "sparse_search_volume": 30,
                    # The higher the number, the more values will be examined with more time.
                    "maj_vote_spk_count": False,
                    # If True, take a majority vote on multiple p-values to estimate the number of speakers.
                    "chunk_cluster_count": 50,
                    # Number of forced clusters (overclustering) per unit chunk in long-form audio clustering.
                    "embeddings_per_chunk": 10000
                    # Number of embeddings in each chunk for long-form audio clustering. Adjust based on GPU memory capacity. (default: 10000, approximately 40 mins of audio)
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
                    "sigmoid_threshold": [0.7],
                    # Sigmoid threshold for generating binarized speaker labels. The smaller the more generous on detecting overlaps.
                    "seq_eval_mode": False,
                    # If True, use oracle number of speaker and evaluate F1 score for the given speaker sequences. Default is False.
                    "split_infer": True,
                    # If True, break the input audio clip to short sequences and calculate cluster average embeddings for inference.
                    "diar_window_length": 50,  # The length of split short sequence when split_infer is True.
                    "overlap_infer_spk_limit": 5,
                    # If the estimated number of speakers are larger than this number, overlap speech is not estimated.

                }
            },
            "enable_msdd": True,  # вмикаємо MSDD
            "convert_to_unique_speaker_ids": True,  # видаватиме speaker_0, speaker_1, ...
        }
    }

    cfg = OmegaConf.create(diar_config)

    return cfg