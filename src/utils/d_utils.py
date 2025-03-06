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
    def weighted_ensemble_speaker_matching(
        cls,
        whisper_segments: List[Dict],
        diar_segments: List[Dict]
    ) -> List[Dict]:
        """Weighted ensemble matching of speakers with transition awareness."""
        final_matched_segments = []
        
        # Define weights for different methods
        method_weights = {
            cls.iou_speaker_matching: 2.0,  # Higher weight for IoU
            cls.probabilistic_speaker_matching: 1.0,
            cls.graph_based_speaker_matching: 1.5
        }
        
        # Add context awareness
        previous_speaker = None
        speaker_change_penalty = 0.3  # Penalty for changing speakers too frequently
        
        for i, wseg in enumerate(whisper_segments):
            speaker_scores = {}
            
            # Apply different methods with weights
            for method, weight in method_weights.items():
                # For IoU method, pass appropriate threshold
                if method == cls.iou_speaker_matching:
                    matched_seg = method([wseg], diar_segments, iou_threshold=0.2)[0]
                else:
                    matched_seg = method([wseg], diar_segments)[0]
                    
                speaker = matched_seg['speaker']
                if speaker != 'UNK':
                    speaker_scores[speaker] = speaker_scores.get(speaker, 0) + weight
            
            # Apply context awareness - favor previous speaker with a small penalty
            if previous_speaker and previous_speaker in speaker_scores:
                # Look at text to detect potential speaker changes
                is_likely_new_speaker = False
                
                # If this segment is very short or contains typical transition phrases
                transition_phrases = ["okay", "thank", "yes", "yeah", "right", "so", "um"]
                if wseg['end'] - wseg['start'] < 1.5 or any(phrase in wseg['text'].lower() for phrase in transition_phrases):
                    is_likely_new_speaker = True
                    
                if not is_likely_new_speaker:
                    speaker_scores[previous_speaker] += speaker_change_penalty
            
            # Choose the speaker with the highest score
            best_speaker = max(speaker_scores, key=speaker_scores.get) if speaker_scores else 'UNK'
            previous_speaker = best_speaker
            
            final_matched_segments.append({
                **wseg,
                'speaker': best_speaker
            })
        
        return final_matched_segments

    @staticmethod
    def improved_merge_adjacent_segments(
            segments: List[Dict],
            max_gap: float = 0.8,  # Reduced from 1.0
            max_segment_duration: float = 8.0,  # Reduced from 10.0
            min_transition_gap: float = 1.5  # New parameter for likely speaker transitions
    ) -> List[Dict]:
        """Improved segment merging with better detection of speaker transitions."""
        if not segments:
            return segments

        merged = [segments[0]]
        for current in segments[1:]:
            prev = merged[-1]
            
            # Conditions for merging
            speaker_match = current['speaker'] == prev['speaker']
            gap = current['start'] - prev['end']
            
            # Check for likely speaker transitions based on text content
            likely_speaker_transition = False
            transition_phrases = ["okay", "thank", "yes", "yeah", "right", "so", "um", "i see", "i understand"]
            
            # If current segment starts with transition phrase or previous ends with one
            if any(prev['text'].lower().strip().endswith(phrase) for phrase in transition_phrases) or \
            any(current['text'].lower().strip().startswith(phrase) for phrase in transition_phrases):
                likely_speaker_transition = True
            
            # Larger gaps are more likely to indicate speaker transitions
            if gap > min_transition_gap:
                likely_speaker_transition = True
                
            # Current duration check
            duration_ok = (current['end'] - current['start']) <= max_segment_duration
            combined_duration_ok = (current['end'] - prev['start']) <= max_segment_duration * 1.5
            
            if speaker_match and gap <= max_gap and duration_ok and combined_duration_ok and not likely_speaker_transition:
                # Extend previous segment
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
            if methods is None:
                methods = [
                DiarizationService.iou_speaker_matching,
                DiarizationService.probabilistic_speaker_matching,
                DiarizationService.graph_based_speaker_matching
            ]
            for func_diarization in methods:
                f.write("FUNCS:\n")
                f.write(f"{func_diarization.__name__}\n")
                f.write(f"{func_diarization.__doc__}\n\n")

            for seg in final_merged:
                line = f"{seg['speaker']} [{seg['start']:.2f}-{seg['end']:.2f}]: {seg['text']}\n"
                f.write(line)
                print(line, end='')
