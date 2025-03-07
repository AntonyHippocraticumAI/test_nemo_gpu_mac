import os
from pydub import AudioSegment

# Путь к исходному файлу:
input_path = f"audio_samples/ravi_inva.wav"

# Папка, куда будут сохранены части (измените на свой путь):
output_folder = f"audio_samples/split_audio_files/ravi_inva"

# Загружаем WAV-файл (pydub автоматически поддерживает формат WAV)
audio = AudioSegment.from_file(input_path, format="wav")

# Определяем длительность файла в миллисекундах
duration_ms = len(audio)
print(duration_ms)

# Вычисляем длительность каждой из 4 частей
part_duration = duration_ms // 4
print(part_duration)

# "Нарезаем" аудио
part1 = audio[0 : part_duration]
part2 = audio[part_duration : 2 * part_duration]
part3 = audio[2 * part_duration : 3 * part_duration]
part4 = audio[3 * part_duration : ]
print(part1, part2, part3, part4)

# Убедимся, что директория существует (создаст, если нет)
os.makedirs(output_folder, exist_ok=True)

# Сохраняем каждую часть в отдельный WAV-файл
part1.export(os.path.join(output_folder, "part1.wav"), format="wav")
part2.export(os.path.join(output_folder, "part2.wav"), format="wav")
part3.export(os.path.join(output_folder, "part3.wav"), format="wav")
part4.export(os.path.join(output_folder, "part4.wav"), format="wav")

print("Разделение завершено и сохранено в:", output_folder)
