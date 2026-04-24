"""Stage 3 benchmark runner and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..session import SessionManifest
from ..utils import write_json
from .inputs import load_stage3_inputs
from .manual import load_manual_speaker_map, validate_manual_speaker_map
from .schemas import resolve_stage3_backend
from .service import execute_stage3


STAGE3_BENCHMARK_BACKENDS = [
    "ollama_decomposed",
    "speechbrain_refmatch",
    "speechbrain_hybrid",
]
RECOMMENDATION_ACCURACY_TIE_THRESHOLD = 2.0
LIGHTER_BACKEND_ORDER = {
    "speechbrain_refmatch": 0,
    "ollama_decomposed": 1,
    "speechbrain_hybrid": 2,
}


def parse_stage3_benchmark_backends(raw_backends: str | None) -> list[str]:
    values = [
        resolve_stage3_backend(value.strip())
        for value in (raw_backends or ",".join(STAGE3_BENCHMARK_BACKENDS)).split(",")
        if value.strip()
    ]
    if not values:
        raise StageExecutionError("Stage 3 benchmark requires at least one backend.")

    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def load_stage3_truth_map(
    *,
    truth_file: Path,
    manifest: SessionManifest,
    stage1_dir: Path,
    stage2_dir: Path,
    participants_file: Path,
) -> dict[str, str]:
    inputs = load_stage3_inputs(
        manifest=manifest,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        participants_file=participants_file,
    )
    speaker_labels = [str(label) for label in inputs.stage2_artifact.get("speaker_labels", [])]
    manual_map = load_manual_speaker_map(truth_file)
    validate_manual_speaker_map(
        manual_map=manual_map,
        speaker_labels=speaker_labels,
        participants=manifest.participants,
        mode="llm",
    )
    missing = sorted(set(speaker_labels) - set(manual_map))
    if missing:
        raise StageExecutionError(
            "Stage 3 benchmark truth speaker map must cover every Stage 2 speaker label. Missing: "
            + ", ".join(missing)
        )
    return {label: manual_map[label] for label in sorted(speaker_labels)}


def score_stage3_assignments(
    *,
    truth_map: dict[str, str],
    speaker_map_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    predicted = {
        str(entry.get("speaker_label")): str(entry.get("assigned_person"))
        for entry in speaker_map_entries
        if entry.get("speaker_label")
    }
    mismatches: list[dict[str, str]] = []
    correct = 0
    for speaker_label, expected_person in sorted(truth_map.items()):
        actual_person = predicted.get(speaker_label)
        if actual_person == expected_person:
            correct += 1
            continue
        mismatches.append(
            {
                "speaker_label": speaker_label,
                "expected_person": expected_person,
                "predicted_person": actual_person or "",
            }
        )
    total = len(truth_map)
    accuracy = round((correct / total) * 100.0, 3) if total else 0.0
    return {
        "correct_assignments": correct,
        "total_assignments": total,
        "exact_assignment_accuracy": accuracy,
        "mismatches": mismatches,
    }


def choose_stage3_benchmark_recommendation(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]
    if not successful:
        return {
            "recommended_backend": None,
            "basis": "no_successful_backends",
            "compared_backends": [],
            "tie_threshold_percentage_points": RECOMMENDATION_ACCURACY_TIE_THRESHOLD,
        }

    ranked = sorted(
        successful,
        key=lambda result: (
            -float(result["exact_assignment_accuracy"]),
            float(result["runtime_seconds"]),
            LIGHTER_BACKEND_ORDER.get(str(result["backend"]), 999),
            str(result["backend"]),
        ),
    )
    best_accuracy = float(ranked[0]["exact_assignment_accuracy"])
    candidates = [
        result
        for result in successful
        if best_accuracy - float(result["exact_assignment_accuracy"]) <= RECOMMENDATION_ACCURACY_TIE_THRESHOLD
    ]
    winner = sorted(
        candidates,
        key=lambda result: (
            float(result["runtime_seconds"]),
            LIGHTER_BACKEND_ORDER.get(str(result["backend"]), 999),
            str(result["backend"]),
        ),
    )[0]
    return {
        "recommended_backend": winner["backend"],
        "basis": (
            "highest_accuracy"
            if len(candidates) == 1
            else "within_accuracy_threshold_prefer_faster_or_lighter"
        ),
        "compared_backends": [result["backend"] for result in ranked],
        "tie_threshold_percentage_points": RECOMMENDATION_ACCURACY_TIE_THRESHOLD,
        "winning_accuracy": winner["exact_assignment_accuracy"],
        "winning_runtime_seconds": winner["runtime_seconds"],
    }


def render_stage3_benchmark_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Stage 3 Benchmark: {report['session_id']}",
        "",
        f"- Truth file: `{report['truth_file']}`",
        f"- Backends: {', '.join(report['backends'])}",
        f"- CPU feasibility notes: {'; '.join(report['cpu_feasibility_notes']) if report['cpu_feasibility_notes'] else 'none'}",
        "",
        "## Recommendation",
        "",
    ]
    recommendation = report["recommendation"]
    if recommendation["recommended_backend"] is None:
        lines.append("- No successful backend runs; no recommendation.")
    else:
        lines.append(
            f"- Recommended backend: `{recommendation['recommended_backend']}` "
            f"({recommendation['basis']})."
        )
        lines.append(
            f"- Winning accuracy: {recommendation['winning_accuracy']:.3f}% in "
            f"{recommendation['winning_runtime_seconds']:.3f}s."
        )

    lines.extend(["", "## Results", ""])
    for result in report["results"]:
        lines.append(f"### {result['backend']}")
        lines.append("")
        lines.append(f"- Status: {result['status']}")
        lines.append(f"- Runtime: {result['runtime_seconds']:.3f}s")
        if result["status"] == "success":
            lines.append(
                f"- Exact assignment accuracy: {result['exact_assignment_accuracy']:.3f}% "
                f"({result['correct_assignments']}/{result['total_assignments']})"
            )
            enrollment = result.get("enrollment_coverage") or {}
            if enrollment:
                lines.append(
                    "- Enrollment coverage: "
                    f"available={len(enrollment.get('available_participants', []))}, "
                    f"required={len(enrollment.get('required_participants', []))}, "
                    f"missing={len(enrollment.get('missing_participants', []))}"
                )
            if result.get("mismatches"):
                lines.append(
                    "- Mismatches: "
                    + "; ".join(
                        f"{item['speaker_label']} expected {item['expected_person']} got {item['predicted_person'] or 'unassigned'}"
                        for item in result["mismatches"]
                    )
                )
        else:
            lines.append(f"- Failure: {result.get('error', 'unknown error')}")
        lines.append(f"- CPU feasibility notes: {'; '.join(result['cpu_feasibility_notes']) or 'none'}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_stage3_benchmark(
    *,
    manifest: SessionManifest,
    stage1_dir: Path,
    stage2_dir: Path,
    runs_dir: Path,
    participants_file: Path,
    truth_file: Path,
    started_at_label: str,
    backends: list[str],
    model: str | None = None,
    cpu_feasibility_notes: list[str] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    truth_map = load_stage3_truth_map(
        truth_file=truth_file,
        manifest=manifest,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        participants_file=participants_file,
    )
    benchmark_root = runs_dir / f"stage3-benchmark.{started_at_label}"
    results: list[dict[str, Any]] = []
    shared_cpu_notes = list(cpu_feasibility_notes or [])

    for backend in backends:
        backend_root = benchmark_root / backend
        stage3_dir = backend_root / "stage3"
        started = perf_counter()
        try:
            output_paths, _, notes, metadata = execute_stage3(
                manifest=manifest,
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                stage3_dir=stage3_dir,
                participants_file=participants_file,
                force=True,
                mode="llm",
                model=model,
                backend=backend,
                speaker_map_path=None,
            )
            elapsed = round(perf_counter() - started, 6)
            artifact_path = stage3_dir / "identified_conversation.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            score = score_stage3_assignments(
                truth_map=truth_map,
                speaker_map_entries=list(artifact.get("speaker_map", [])),
            )
            backend_usage = metadata.get("backend_usage")
            results.append(
                {
                    "backend": backend,
                    "status": "success",
                    "runtime_seconds": elapsed,
                    "step_runtimes": [{"step": "execute_stage3", "elapsed_seconds": elapsed}],
                    "output_paths": output_paths,
                    "notes": notes,
                    "backend_usage": backend_usage,
                    "llm_usage": metadata.get("llm_usage"),
                    "enrollment_coverage": (backend_usage or {}).get("enrollment_coverage"),
                    "cpu_feasibility_notes": list(shared_cpu_notes),
                    **score,
                }
            )
        except StageExecutionError as exc:
            elapsed = round(perf_counter() - started, 6)
            results.append(
                {
                    "backend": backend,
                    "status": "failed",
                    "runtime_seconds": elapsed,
                    "step_runtimes": [{"step": "execute_stage3", "elapsed_seconds": elapsed}],
                    "error": str(exc),
                    "notes": [str(exc)],
                    "backend_usage": None,
                    "llm_usage": None,
                    "enrollment_coverage": None,
                    "cpu_feasibility_notes": list(shared_cpu_notes),
                }
            )
        except Exception as exc:
            elapsed = round(perf_counter() - started, 6)
            results.append(
                {
                    "backend": backend,
                    "status": "failed",
                    "runtime_seconds": elapsed,
                    "step_runtimes": [{"step": "execute_stage3", "elapsed_seconds": elapsed}],
                    "error": f"Unexpected error: {exc}",
                    "notes": [f"Unexpected error: {exc}"],
                    "backend_usage": None,
                    "llm_usage": None,
                    "enrollment_coverage": None,
                    "cpu_feasibility_notes": list(shared_cpu_notes),
                }
            )

    recommendation = choose_stage3_benchmark_recommendation(results)
    report = {
        "stage": "stage3_benchmark",
        "session_id": manifest.session_id,
        "truth_file": repo_relative(truth_file),
        "participants_file": repo_relative(participants_file),
        "backends": backends,
        "cpu_feasibility_notes": shared_cpu_notes,
        "benchmark_root": repo_relative(benchmark_root),
        "results": results,
        "recommendation": recommendation,
    }
    json_path = runs_dir / f"stage3-benchmark.{started_at_label}.json"
    markdown_path = runs_dir / f"stage3-benchmark.{started_at_label}.md"
    write_json(json_path, report)
    markdown_path.write_text(render_stage3_benchmark_markdown(report), encoding="utf-8")
    return report, json_path, markdown_path
