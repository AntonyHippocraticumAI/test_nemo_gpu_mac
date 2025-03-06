import json


def create_manifest(audio_path, manifest_path, force_speakers):
    # 1) Створюємо маніфест (JSON) на одне аудіо:
    manifest_entry = {
        "audio_filepath": audio_path,
        "offset": 0,
        "duration": None,
        "label": "infer",  # умовно
        "text": "-",
        "num_speakers": force_speakers,  # Якщо точно 2, пишемо 2
        "rttm_filepath": None,
        "uem_filepath": None
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_entry) + "\n")