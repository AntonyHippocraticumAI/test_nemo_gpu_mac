import nemo.collections.asr as nemo_asr

# Вказуємо шлях до файлу моделі
vad_model_path = "/Users/antonandreev/commandrecognition_en_matchboxnet3x2x64_v2_subset_task_v1.0.0rc1/commandrecognition_en_matchboxnet3x2x64_v2_subset_task.nemo"

# Завантажуємо модель локально
vad_model = nemo_asr.models.EncDecClassificationModel.restore_from(vad_model_path)

# Перевіряємо завантаження
print(vad_model)
