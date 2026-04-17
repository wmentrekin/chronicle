"""Standalone SpeechBrain-style spike runner for the separate Stage 2 runtime."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--vad-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    return parser.parse_args()


def merge_labeled_segments(
    segments: list[dict[str, float | str]],
    *,
    join_gap_seconds: float = 0.15,
) -> list[dict[str, float | str]]:
    merged: list[dict[str, float | str]] = []
    for segment in sorted(segments, key=lambda item: (float(item["start"]), float(item["end"]))):
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        if (
            previous["speaker_label"] == segment["speaker_label"]
            and float(segment["start"]) <= float(previous["end"]) + join_gap_seconds
        ):
            previous["end"] = max(float(previous["end"]), float(segment["end"]))
            continue
        merged.append(segment)
    return merged


def build_non_overlapping_turns(
    segments: list[dict[str, float | str]],
    speech_regions: list[dict[str, float]],
) -> list[dict[str, float | str]]:
    if not segments:
        return []

    candidate_boundaries = {
        round(float(segment["start"]), 3)
        for segment in segments
    } | {
        round(float(segment["end"]), 3)
        for segment in segments
    } | {
        round(float(region["start"]), 3)
        for region in speech_regions
    } | {
        round(float(region["end"]), 3)
        for region in speech_regions
    }
    boundaries = sorted(candidate_boundaries)

    intervals: list[dict[str, float | str]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left < 0.05:
            continue
        midpoint = (left + right) / 2.0
        if not any(float(region["start"]) <= midpoint <= float(region["end"]) for region in speech_regions):
            continue

        active = [
            segment
            for segment in segments
            if float(segment["start"]) <= midpoint < float(segment["end"])
        ]
        if not active:
            continue

        label_scores: dict[str, tuple[int, float, float]] = {}
        for segment in active:
            label = str(segment["speaker_label"])
            count, total_duration, center_distance = label_scores.get(label, (0, 0.0, 0.0))
            duration = float(segment["end"]) - float(segment["start"])
            segment_midpoint = (float(segment["start"]) + float(segment["end"])) / 2.0
            label_scores[label] = (
                count + 1,
                total_duration + duration,
                center_distance - abs(segment_midpoint - midpoint),
            )

        selected_label = max(
            label_scores.items(),
            key=lambda item: (item[1][0], item[1][1], item[1][2], item[0]),
        )[0]
        intervals.append(
            {
                "speaker_label": selected_label,
                "start": round(left, 3),
                "end": round(right, 3),
            }
        )

    merged = merge_labeled_segments(intervals, join_gap_seconds=0.05)

    smoothed: list[dict[str, float | str]] = []
    for segment in merged:
        duration = float(segment["end"]) - float(segment["start"])
        if duration >= 0.3 or not smoothed:
            smoothed.append(segment)
            continue

        previous = smoothed[-1]
        previous_duration = float(previous["end"]) - float(previous["start"])
        if previous_duration >= duration:
            previous["end"] = max(float(previous["end"]), float(segment["end"]))
        else:
            segment["start"] = previous["start"]
            smoothed[-1] = segment

    return merge_labeled_segments(smoothed, join_gap_seconds=0.05)


def main() -> None:
    args = parse_args()

    import torch
    import torchaudio
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    try:
        from speechbrain.inference.VAD import VAD
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier, VAD

    audio_path = Path(args.audio)

    started = time.perf_counter()
    vad = VAD.from_hparams(
        source=args.vad_model,
        run_opts={"device": args.device},
    )
    embedder = EncoderClassifier.from_hparams(
        source=args.embedding_model,
        run_opts={"device": args.device},
    )
    load_seconds = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    speech_boundaries = vad.get_speech_segments(audio_path.as_posix())
    waveform, sample_rate = torchaudio.load(audio_path.as_posix())
    waveform = waveform.mean(dim=0, keepdim=True)

    subsegments: list[dict[str, float | np.ndarray]] = []
    speech_regions: list[dict[str, float]] = []

    for boundary in speech_boundaries:
        start = float(boundary[0])
        end = float(boundary[1])
        speech_regions.append({"start": round(start, 3), "end": round(end, 3)})
        duration = end - start
        if duration < 0.5:
            continue

        if duration <= 2.0:
            windows = [(start, end)]
        else:
            windows = []
            cursor = start
            while cursor < end:
                window_end = min(end, cursor + 1.5)
                if window_end - cursor >= 0.5:
                    windows.append((cursor, window_end))
                if window_end >= end:
                    break
                cursor += 0.75

        for win_start, win_end in windows:
            start_sample = int(win_start * sample_rate)
            end_sample = int(win_end * sample_rate)
            chunk = waveform[:, start_sample:end_sample]
            if chunk.numel() == 0:
                continue
            with torch.no_grad():
                embedding = embedder.encode_batch(chunk).squeeze().detach().cpu().numpy()
            subsegments.append(
                {
                    "start": round(win_start, 3),
                    "end": round(win_end, 3),
                    "embedding": embedding,
                }
            )

    if not subsegments:
        payload = {
            "backend": "speechbrain",
            "device": args.device,
            "load_seconds": load_seconds,
            "run_seconds": round(time.perf_counter() - started, 2),
            "turn_count": 0,
            "speakers": [],
            "speech_region_count": len(speech_regions),
            "subsegment_count": 0,
            "turns": [],
        }
        sys.stdout.write(json.dumps(payload))
        return

    embeddings = np.stack([segment["embedding"] for segment in subsegments])
    deltas = np.diff(embeddings, axis=0)
    embedding_norm_mean = float(np.linalg.norm(deltas, axis=1).mean()) if len(deltas) else 0.0

    if args.num_speakers is not None:
        cluster_count = args.num_speakers
    else:
        min_speakers = args.min_speakers or 2
        max_speakers = args.max_speakers or min(6, len(subsegments))
        max_speakers = max(min_speakers, min(max_speakers, len(subsegments)))

        best_score = None
        best_cluster_count = min_speakers
        for candidate in range(min_speakers, max_speakers + 1):
            if candidate >= len(subsegments):
                break
            clustering = AgglomerativeClustering(
                n_clusters=candidate,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)
            if len(set(labels)) <= 1:
                continue
            score = silhouette_score(embeddings, labels, metric="cosine")
            if best_score is None or score > best_score:
                best_score = score
                best_cluster_count = candidate
        cluster_count = best_cluster_count

    clustering = AgglomerativeClustering(
        n_clusters=min(cluster_count, len(subsegments)),
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    labeled_segments = []
    for index, (segment, label) in enumerate(zip(subsegments, labels), start=1):
        labeled_segments.append(
            {
                "turn_id": index,
                "speaker_label": f"SPEAKER_{label:02d}",
                "start": float(segment["start"]),
                "end": float(segment["end"]),
            }
        )

    merged = build_non_overlapping_turns(labeled_segments, speech_regions)
    turns = []
    for index, segment in enumerate(merged, start=1):
        turns.append(
            {
                "turn_id": index,
                "speaker_label": segment["speaker_label"],
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
            }
        )

    run_seconds = round(time.perf_counter() - started, 2)
    payload = {
        "backend": "speechbrain",
        "device": args.device,
        "load_seconds": load_seconds,
        "run_seconds": run_seconds,
        "turn_count": len(turns),
        "speakers": sorted({turn["speaker_label"] for turn in turns}),
        "speech_region_count": len(speech_regions),
        "subsegment_count": len(subsegments),
        "estimated_embedding_delta_norm_mean": round(embedding_norm_mean, 4),
        "raw_labeled_subsegment_count": len(labeled_segments),
        "turns": turns,
    }
    sys.stdout.write(json.dumps(payload))


if __name__ == "__main__":
    main()
