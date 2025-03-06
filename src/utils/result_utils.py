import os
from datetime import datetime

def generate_readable_filename(result_dir: str) -> str:
    base_name, ext = os.path.splitext(result_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d---%H:%M:%S")

    return f"{base_name}{timestamp}{ext}"


def split_by_words(text, words_per_line=10):
    words = text.split()
    lines = [" ".join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
    return "\n".join(lines)