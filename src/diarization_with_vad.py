import os
import tempfile
import librosa
import numpy as np
import torch
import soundfile as sf

# -------- FASTER-WHISPER -----------
from faster_whisper import WhisperModel, BatchedInferencePipeline

# -------- NEMO (VAD + Speaker) ---
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.asr.models import EncDecSpeakerLabelModel
import torch.nn.functional as F

# -------- КЛАСТЕРИЗАЦІЯ -------------
from sklearn.cluster import AgglomerativeClustering

##############################################################################
# НАЛАШТУВАННЯ
##############################################################################
AUDIO_PATH = "/Users/antonandreev/python_prog/test_nemo_cpu_mac/audio_samples/recording_16000hz.wav"
WHISPER_MODEL_NAME = "large-v3"  # "large-v2", "medium", і т.п.
FORCE_SPEAKERS = None  # Спробуйте: None, або 2, або 3...
DISTANCE_THRESHOLD = 0.7  # Якщо FORCE_SPEAKERS=None, алгоритм візьме цей поріг
VAD_MODEL_PATH = "/Users/antonandreev/python_prog/test_nemo_cpu_mac/vad/vad_marblenet.nemo"  # Або інший
SPEAKER_MODEL_PATH = "/Users/antonandreev/speakerverification_speakernet_v1.6.0/speakerverification_speakernet.nemo"               # Titanet model
##############################################################################


def step_a_transcribe_with_batched_pipeline(audio: np.ndarray,
                                            sr: int = 16000,
                                            whisper_model_name: str = "large-v3",
                                            batch_size: int = 8):
    """
    КРОК A: Транскрибуємо повне аудіо одним викликом BatchedInferencePipeline.
    Повертаємо список [{'start':..., 'end':..., 'text':...}, ...].
    """
    print("\n=== STEP A: Transcribe whole audio with BatchedInferencePipeline ===")
    whisper_model = WhisperModel(
        whisper_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8",
        num_workers=4,
    )
    pipeline = BatchedInferencePipeline(whisper_model)

    segments, info = pipeline.transcribe(
        audio,
        batch_size=batch_size,
        beam_size=1,
        without_timestamps=False,  # треба, щоб отримати start/end
    )

    whisper_segments = []
    for seg in segments:
        whisper_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })

    print(f"Total segments from Whisper: {len(whisper_segments)}")
    return whisper_segments


def step_b_speaker_diarization(audio_path: str,
                               vad_model_path: str,
                               speaker_model_path: str,
                               shift_length: float = 0.03,
                               embedding_window: float = 1.5,
                               embedding_shift: float = 0.75,
                               force_speakers=None,
                               distance_threshold=0.7):
    """
    КРОК B:
    1) ВАД на дрібних кроках (shift_length),
    2) Отримання ембедінгів Titanet на "вікнах" (embedding_window),
    3) Кластеризація AgglomerativeClustering.
       - Якщо force_speakers = <число>, то n_clusters = <число>.
       - Якщо force_speakers = None, то n_clusters=None + distance_threshold.

    Повертаємо: список [{'speaker':..., 'start':..., 'end':...}, ...].
    """
    print("\n=== STEP B: Speaker Diarization (VAD + Titanet + clustering) ===")

    # 1) Завантажимо аудіо
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio loaded: {audio_path}, duration = {len(audio)/sr:.2f} s")

    # 2) VAD-модель
    print(f"Loading VAD model from: {vad_model_path}")
    vad_model = EncDecClassificationModel.restore_from(vad_model_path)
    vad_model.eval()
    if torch.cuda.is_available():
        vad_model.cuda()
    print("VAD loaded")

    # 3) Крокове VAD (просте)
    shift_samples = int(shift_length * sr)
    speech_probs = []
    idx = 0
    window_samples = shift_samples

    while idx + window_samples <= len(audio):
        chunk = audio[idx: idx + window_samples]
        chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        if torch.cuda.is_available():
            chunk_tensor = chunk_tensor.cuda()

        with torch.no_grad():
            logits = vad_model(
                input_signal=chunk_tensor,
                input_signal_length=torch.tensor([chunk_tensor.shape[1]])
            )
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        speech_prob = probs[1]  # індекс 1 = speech
        speech_probs.append((idx, speech_prob))
        idx += shift_samples

    diar_frames = []
    is_speech = False
    seg_start = 0.0

    for (sample_i, prob) in speech_probs:
        t_sec = sample_i / sr
        if not is_speech and prob > 0.5:
            is_speech = True
            seg_start = t_sec
        elif is_speech and prob <= 0.5:
            is_speech = False
            seg_end = t_sec
            diar_frames.append({"start": seg_start, "end": seg_end})

    if is_speech:
        diar_frames.append({"start": seg_start, "end": len(audio)/sr})

    print(f"Found {len(diar_frames)} raw VAD segments")

    # 4) Завантажуємо Titanet
    print(f"Loading Speaker model from: {speaker_model_path}")
    speaker_model = EncDecSpeakerLabelModel.restore_from(speaker_model_path)
    speaker_model.eval()
    if torch.cuda.is_available():
        speaker_model.cuda()
    print("Titanet loaded")

    # 5) Отримуємо ембедінги для вікон ~1.5 s
    emb_window_samples = int(embedding_window * sr)
    emb_shift_samples = int(embedding_shift * sr)

    embedding_results = []

    for seg in diar_frames:
        seg_start_s = seg["start"]
        seg_end_s = seg["end"]
        seg_start_i = int(seg_start_s * sr)
        seg_end_i = int(seg_end_s * sr)

        if (seg_end_s - seg_start_s) < 0.1:
            continue

        idx = seg_start_i
        while idx < seg_end_i:
            chunk_end_i = idx + emb_window_samples
            if chunk_end_i > seg_end_i:
                chunk_end_i = seg_end_i

            chunk = audio[idx:chunk_end_i]

            # Пишемо у тимчасовий .wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmp_wav_path = tmpfile.name
            sf.write(tmp_wav_path, chunk, sr)

            with torch.no_grad():
                emb = speaker_model.get_embedding(path2audio_file=tmp_wav_path)
            os.remove(tmp_wav_path)

            # "emb" може бути torch.Tensor shape=(1, emb_dim)
            emb_np = emb.cpu().numpy()[0]

            emb_start_sec = idx / sr
            emb_end_sec = chunk_end_i / sr
            embedding_results.append({
                "start": emb_start_sec,
                "end": emb_end_sec,
                "embedding": emb_np
            })

            idx += emb_shift_samples

    print(f"Created {len(embedding_results)} embedding windows")

    if len(embedding_results) == 0:
        print("No embeddings => no speech or too short segments.")
        return []

    # 6) Кластеризація
    X = np.array([r["embedding"] for r in embedding_results])

    if force_speakers is not None:
        # Примусово
        print(f"Using forced n_clusters={force_speakers}")
        clustering = AgglomerativeClustering(
            n_clusters=force_speakers,
            linkage="average"
        )
    else:
        # Автовизначення
        print(f"Using distance_threshold={distance_threshold} (auto num. clusters)")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="average"
        )

    labels = clustering.fit_predict(X)

    for i, lab in enumerate(labels):
        embedding_results[i]["speaker_label"] = lab
        print("EMBEDING", embedding_results[i]["speaker_label"])

    n_speakers = labels.max() + 1
    print(f"Number of speakers found: {n_speakers}")

    # 7) Формуємо фінальні діаризаційні сегменти (поки без злиття сусідніх)
    diarization_segments = []
    for emb_res in embedding_results:
        diarization_segments.append({
            "speaker": emb_res["speaker_label"],
            "start": emb_res["start"],
            "end": emb_res["end"]
        })

    # Сортуємо за часом
    diarization_segments.sort(key=lambda x: x["start"])
    return diarization_segments


def merge_diarization_with_whisper(whisper_segments: list, diar_segments: list):
    """
    Зливаємо за часом:
      whisper_segments = [{"start":..., "end":..., "text":...}, ...]
      diar_segments    = [{"start":..., "end":..., "speaker":...}, ...]
    Повертаємо [{"start":..., "end":..., "speaker":..., "text":...}, ...].
    """
    i, j = 0, 0
    merged = []

    while i < len(whisper_segments) and j < len(diar_segments):
        wseg = whisper_segments[i]
        dseg = diar_segments[j]

        overlap_start = max(wseg["start"], dseg["start"])
        overlap_end = min(wseg["end"], dseg["end"])

        if overlap_end > overlap_start:
            merged.append({
                "start": overlap_start,
                "end": overlap_end,
                "speaker": dseg["speaker"],
                "text": wseg["text"]
            })

        # Хто раніше закінчився
        if wseg["end"] < dseg["end"]:
            i += 1
        else:
            j += 1

    return merged


def main():
    # 1) Зчитуємо аудіо (для Whisper)
    audio, sr = librosa.load(AUDIO_PATH, sr=16000)

    # 2) Крок A: Транскрибувати одним викликом (BatchedInferencePipeline)
    whisper_segments = step_a_transcribe_with_batched_pipeline(
        audio, sr=sr,
        whisper_model_name=WHISPER_MODEL_NAME,
        batch_size=8
    )

    # 3) Крок B: Спікер-діаризація (VAD + Titanet + clustering)
    diar_segments = step_b_speaker_diarization(
        audio_path=AUDIO_PATH,
        vad_model_path=VAD_MODEL_PATH,
        speaker_model_path=SPEAKER_MODEL_PATH,
        shift_length=0.03,
        embedding_window=1.5,
        embedding_shift=0.75,
        force_speakers=FORCE_SPEAKERS,        # <--- можна задати int або None
        distance_threshold=DISTANCE_THRESHOLD # <--- якщо force_speakers=None
    )

    # 4) Злиття
    merged_segments = merge_diarization_with_whisper(whisper_segments, diar_segments)

    # 5) Друк
    print("\n=== FINAL MERGED OUTPUT ===")
    for seg in merged_segments:
        spk = seg["speaker"]
        st, en = seg["start"], seg["end"]
        txt = seg["text"].replace("\n", " ").strip()
        print(f"Speaker {spk} [{st:.2f}-{en:.2f}]: {txt}")


if __name__ == "__main__":
    main()
