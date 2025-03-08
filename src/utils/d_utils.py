import os
import typing
from pathlib import Path

import numpy as np
import networkx as nx
from typing import List, Dict, Any

from utils.result_utils import generate_readable_filename


class DiarizationService:
    @staticmethod
    def calculate_overlap(seg1: Dict, seg2: Dict) -> float:
        """Обчислення перекриття між двома сегментами."""
        start = max(seg1['start'], seg2['start'])
        end = min(seg1['end'], seg2['end'])
        return max(0, end - start)

    @staticmethod
    def intersection_over_union(wseg: Dict, dseg: Dict) -> float:
        """Обчислення IoU для сегментів."""
        intersection = DiarizationService.calculate_overlap(wseg, dseg)
        union = (wseg['end'] - wseg['start']) + (dseg['end'] - dseg['start']) - intersection
        return intersection / union if union > 0 else 0

    @staticmethod
    def segment_distance(wseg: Dict, dseg: Dict) -> float:
        """Обчислення відстані між сегментами."""
        time_overlap = DiarizationService.calculate_overlap(wseg, dseg)
        time_distance = abs(wseg['start'] - dseg['start'])
        duration_diff = abs((wseg['end'] - wseg['start']) - (dseg['end'] - dseg['start']))
        return time_distance + duration_diff - time_overlap

    @classmethod
    def iou_speaker_matching(
            cls,
            whisper_segments: List[Dict],
            diar_segments: List[Dict],
            iou_threshold: float = 0.3
    ) -> List[Dict]:
        """Matching speakers based on IoU."""
        matched_segments = []
        for wseg in whisper_segments:
            best_match = None
            best_iou = 0
            for dseg in diar_segments:
                iou = cls.intersection_over_union(wseg, dseg)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = dseg['speaker']

            matched_segments.append({
                **wseg,
                'speaker': best_match if best_match else 'UNK'
            })
        return matched_segments

    @classmethod
    def probabilistic_speaker_matching(
            cls,
            whisper_segments: List[Dict],
            diar_segments: List[Dict]
    ) -> List[Dict]:
        """Probabilistic matching of speakers"""
        matched_segments = []
        for wseg in whisper_segments:
            distances = [cls.segment_distance(wseg, dseg) for dseg in diar_segments]
            best_match_idx = np.argmin(distances)

            matched_segments.append({
                **wseg,
                'speaker': diar_segments[best_match_idx]['speaker']
            })
        return matched_segments

    @classmethod
    def graph_based_speaker_matching(
            cls,
            whisper_segments: List[Dict],
            diar_segments: List[Dict]
    ) -> List[Dict]:
        """Graph matching of speakers."""
        G = nx.Graph()

        # Додавання вершин
        for i, wseg in enumerate(whisper_segments):
            G.add_node(f'whisper_{i}', type='whisper', segment=wseg)

        for j, dseg in enumerate(diar_segments):
            G.add_node(f'diar_{j}', type='diar', segment=dseg)

        # Додавання ребер на основі перекриття
        for i, wseg in enumerate(whisper_segments):
            for j, dseg in enumerate(diar_segments):
                overlap = cls.calculate_overlap(wseg, dseg)
                if overlap > 0:
                    G.add_edge(f'whisper_{i}', f'diar_{j}', weight=overlap)

        # Знаходження максимального паронування
        matching = nx.max_weight_matching(G)

        # Перетворення результатів
        matched_segments = []
        for i, wseg in enumerate(whisper_segments):
            speaker = 'UNK'
            for match in matching:
                if f'whisper_{i}' in match:
                    partner = [m for m in match if 'diar_' in m][0]
                    speaker = diar_segments[int(partner.split('_')[1])]['speaker']
                    break

            matched_segments.append({
                **wseg,
                'speaker': speaker
            })

        return matched_segments

    @classmethod
    def ensemble_speaker_matching(
            cls,
            whisper_segments: List[Dict],
            diar_segments: List[Dict],
            methods: List[typing.Callable] = None
    ) -> List[Dict]:
        """Ensemble-метод зіставлення спікерів."""
        if methods is None:
            methods = [
                # cls.iou_speaker_matching,
                cls.probabilistic_speaker_matching,
                # cls.graph_based_speaker_matching
            ]

        final_matched_segments = []

        for wseg in whisper_segments:
            speaker_votes = {}

            # Застосування різних методів
            for method in methods:
                matched_seg = method([wseg], diar_segments)[0]
                speaker = matched_seg['speaker']
                if speaker != 'UNK':
                    speaker_votes[speaker] = speaker_votes.get(speaker, 0) + 1

            # Вибір спікера з найбільшою кількістю голосів
            best_speaker = max(speaker_votes, key=speaker_votes.get) if speaker_votes else 'UNK'

            final_matched_segments.append({
                **wseg,
                'speaker': best_speaker
            })

        return final_matched_segments

    @staticmethod
    def merge_adjacent_segments(
            segments: List[Dict],
            max_gap: float = 1.0,
            max_segment_duration: float = 10.0
    ) -> List[Dict]:
        """Злиття суміжних сегментів."""
        if not segments:
            return segments

        merged = [segments[0]]
        for current in segments[1:]:
            prev = merged[-1]

            # Умови для злиття
            speaker_match = current['speaker'] == prev['speaker']
            gap_small = current['start'] - prev['end'] <= max_gap
            duration_ok = (current['end'] - current['start']) <= max_segment_duration

            if speaker_match and gap_small and duration_ok:
                # Розширюємо попередній сегмент
                prev['end'] = current['end']
                prev['text'] += ' ' + current['text']
            else:
                merged.append(current)

        return merged

    @staticmethod
    def export_results(
            final_merged: List[Dict],
            methods: List
    ) -> None:
        """Експорт результатів діаризації."""

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "segmentations_and_transcriptions/")
        RESULT_FILE_NAME = generate_readable_filename(RESULT_DIR)

        os.makedirs(os.path.dirname(RESULT_FILE_NAME), exist_ok=True)

        with open(f"{RESULT_FILE_NAME}.txt", 'w', encoding='utf-8') as f:
            # for func_diarization in methods:
            #     f.write("FUNCS:\n")
            #     f.write(f"{func_diarization.__name__}\n")
            #     f.write(f"{func_diarization.__doc__}\n\n")

            for seg in final_merged:
                line = f"{seg['speaker']} [{seg['start']:.2f}-{seg['end']:.2f}]: {seg['text']}\n"
                f.write(line)
                print(line, end='')
    


    @staticmethod
    def word_level_alignment(whisper_segments: List[Dict], diar_segments: List[Dict]) -> List[Dict]:
        """
        Пробегаемся по всем словам из Whisper и сопоставляем их с интервами NeMo.
        Возвращает список сегментов (start, end, text, speaker), уже «разрезанных» по говорящему.
        """

        # Предполагаем, что diar_segments = [{start, end, speaker}, ...], 
        # отсортировано по start.
        if not diar_segments:
            return []

        # Список финальных кусков
        results = []

        # Индекс NeMo-интервала
        nemo_idx = 0
        total_diar = len(diar_segments)
        cur_diar = diar_segments[nemo_idx]

        for seg in whisper_segments:
            for w in seg.get("words", []):
                w_start = w["start"]
                w_end   = w["end"]

                # Двигаем индекс NeMo, пока текущее слово не попадет в нужный интервал
                while nemo_idx < total_diar and w_start >= cur_diar["end"]:
                    # перешли к следующему интервалу
                    nemo_idx += 1
                    if nemo_idx < total_diar:
                        cur_diar = diar_segments[nemo_idx]
                    else:
                        break

                if nemo_idx >= total_diar:
                    # Вышли за все интервалы NeMo
                    break

                # Теперь проверяем, пересекается ли слово с cur_diar
                if w_end <= cur_diar["end"]:
                    # Полностью лежит в этом интервале
                    results.append({
                        "speaker": cur_diar["speaker"],
                        "start": w_start,
                        "end": w_end,
                        "text": w["word"]
                    })
                else:
                    # Считаем, где оно «длиннее»:
                    overlap_with_current = cur_diar["end"] - w_start  # часть слова внутри этого интервала
                    word_duration = w_end - w_start

                    # overlap_with_next = (w_end - cur_diar["end"])  # часть слова внутри следующего интервала
                    # Однако, если у нас идет цепочка интервалов, нужно аккуратно взять следующий индекс (nemo_idx+1),
                    # и проверить, не вышли ли мы за пределы массива diar_segments.

                    # Для простоты — если overlap_with_current >= (word_duration / 2),
                    # то считаем слово целиком текущему спикеру, иначе — следующему
                    if overlap_with_current >= (word_duration / 2):
                        # отдать слово текущему спикеру
                        results.append({
                            "speaker": cur_diar["speaker"],
                            "start": w_start,
                            "end": w_end,
                            "text": w["word"]
                        })
                    else:
                        # переходим к следующему интервалу
                        nemo_idx += 1
                        if nemo_idx < total_diar:
                            cur_diar = diar_segments[nemo_idx]
                            # отдаем слово уже туда
                            results.append({
                                "speaker": cur_diar["speaker"],
                                "start": w_start,
                                "end": w_end,
                                "text": w["word"]
                            })
                        else:
                            # уже нет интервалов
                            results.append({
                                "speaker": "UNK",
                                "start": w_start,
                                "end": w_end,
                                "text": w["word"]
                            })
        
        return results



    @staticmethod
    def merge_word_level_results(word_level_segments: List[Dict], max_silence: float = 0.8) -> List[Dict]:
        """
        Принимает список {speaker, start, end, text} (каждый ~ одно слово)
        и склеивает подряд идущие слова одного спикера, если пауза между 
        ними меньше max_silence.
        Возвращает список более крупных сегментов.
        """
        if not word_level_segments:
            return []

        merged = []
        prev = None

        for w in word_level_segments:
            if prev is None:
                prev = {
                    "speaker": w["speaker"],
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["text"]
                }
                continue

            # Если тот же спикер и небольшая пауза
            if (w["speaker"] == prev["speaker"]) and (w["start"] - prev["end"] <= max_silence):
                # "дописываем" слово
                prev["end"] = w["end"]
                prev["text"] = prev["text"] + " " + w["text"]
            else:
                # заканчиваем предыдущий, начинаем новый
                merged.append(prev)
                prev = {
                    "speaker": w["speaker"],
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["text"]
                }

        if prev:
            merged.append(prev)

        return merged


    