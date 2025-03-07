import os

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class RttmUtils:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.rttm_path = None


    def check_the_results_of_diarization(self, file_name: str):
        rttm_path = os.path.join(f"nemo_diar_temp/diar_output/pred_rttms/{file_name}")
        if not os.path.isfile(rttm_path):
            logger.error(f"RTTM not found at {rttm_path}. Diarization failed or config error.")
            raise

        self.rttm_path = rttm_path
        return self.rttm_path


    def format_diarization_rttm(self):
        diar_segments = []
        with open(self.rttm_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            # Пропустимо неправильні строки
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start_time = float(parts[3])
            duration = float(parts[4])
            spk_label = parts[7]
            diar_segments.append({
                "start": start_time,
                "end":   start_time + duration,
                "speaker": spk_label
            })

        # Сортуємо за часом
        diar_segments.sort(key=lambda x: x["start"])

        return diar_segments
