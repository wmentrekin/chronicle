"""Standalone pyannote spike runner for the separate Stage 2 runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required in the environment for the pyannote spike.")

    from pyannote.audio import Pipeline

    started = time.perf_counter()
    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=token)
    load_seconds = round(time.perf_counter() - started, 2)

    if args.device != "cpu":
        import torch

        pipeline.to(torch.device(args.device))

    diarize_kwargs: dict[str, int] = {}
    if args.num_speakers is not None:
        diarize_kwargs["num_speakers"] = args.num_speakers
    if args.min_speakers is not None:
        diarize_kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        diarize_kwargs["max_speakers"] = args.max_speakers

    started = time.perf_counter()
    diarization = pipeline(Path(args.audio).as_posix(), **diarize_kwargs)
    run_seconds = round(time.perf_counter() - started, 2)

    turns = []
    for index, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True), start=1):
        turns.append(
            {
                "turn_id": index,
                "speaker_label": speaker,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
            }
        )

    payload = {
        "model": args.model,
        "device": args.device,
        "load_seconds": load_seconds,
        "run_seconds": run_seconds,
        "turn_count": len(turns),
        "speakers": sorted({turn["speaker_label"] for turn in turns}),
        "turns": turns,
    }
    sys.stdout.write(json.dumps(payload))


if __name__ == "__main__":
    main()
