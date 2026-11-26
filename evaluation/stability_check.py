"""Repeat selected RAG generations to measure answer stability."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from backend.config import config
from evaluation.rag_runner import run_rag


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = PROJECT_ROOT / "evaluation" / "testset.json"
BASELINE_OUTPUTS_PATH = PROJECT_ROOT / "evaluation" / "cache" / "baseline" / "outputs.jsonl"
V2_OUTPUTS_PATH = (
    PROJECT_ROOT / "evaluation" / "cache" / "baseline-v2" / "outputs.jsonl"
)
V2_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "baseline-v2" / "summary.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation" / "cache" / "stability" / "outputs.jsonl"
)
NAMESPACE = "eval-baseline-v2"
ATTEMPTS_PER_ROW = 3
SLEEP_SECONDS = 1.0
REFUSAL_PREFIX = "i don't know"


def _read_jsonl(path: Path) -> Dict[str, Dict]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"Invalid record at {path}:{line_number}.")
            records[str(record["id"])] = record
    return records


def _read_attempts(path: Path) -> Tuple[List[Dict], set]:
    records = []
    completed = set()
    if not path.exists():
        return records, completed
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record at {path}:{line_number}.")
            key = (str(record.get("id", "")), int(record.get("attempt", 0)))
            if not key[0] or key[1] < 1 or key in completed:
                raise ValueError(f"Invalid or duplicate attempt at {path}:{line_number}.")
            completed.add(key)
            records.append(record)
    return records, completed


def _write_attempts(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def _abstained(output: Dict) -> bool:
    answer = output.get("answer")
    return isinstance(answer, str) and answer.lstrip().casefold().startswith(
        REFUSAL_PREFIX
    )


def _select_rows() -> Tuple[List[Dict], List[Dict]]:
    rows = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    baseline_outputs = _read_jsonl(BASELINE_OUTPUTS_PATH)
    v2_outputs = _read_jsonl(V2_OUTPUTS_PATH)
    if not baseline_outputs or not v2_outputs:
        raise RuntimeError("Both baseline output caches are required.")

    answerable = [
        row
        for row in rows
        if row.get("type") != "unanswerable"
        and str(row["id"]) in baseline_outputs
        and str(row["id"]) in v2_outputs
    ]
    previously_abstained = [
        row for row in answerable if _abstained(v2_outputs[str(row["id"])])
    ]
    wanted_by_type = Counter(row["type"] for row in previously_abstained)

    controls = []
    for row in answerable:
        row_id = str(row["id"])
        if (
            wanted_by_type[row["type"]] > 0
            and not _abstained(baseline_outputs[row_id])
            and not _abstained(v2_outputs[row_id])
        ):
            controls.append(row)
            wanted_by_type[row["type"]] -= 1

    if len(previously_abstained) != 10 or len(controls) != 10 or any(
        wanted_by_type.values()
    ):
        raise RuntimeError(
            "Expected 10 baseline-v2 abstainers and 10 type-matched controls."
        )
    return previously_abstained, controls


def main() -> None:
    summary = json.loads(V2_SUMMARY_PATH.read_text(encoding="utf-8"))
    top_k = int(summary["config"]["top_k"])
    primary_generator = config.GROQ_MODEL.strip()
    previously_abstained, controls = _select_rows()
    selected_rows = previously_abstained + controls

    existing_records, completed = _read_attempts(OUTPUT_PATH)
    records_by_key = {
        (str(record["id"]), int(record["attempt"])): record
        for record in existing_records
    }

    print(f"Primary generator: {primary_generator}")
    print(f"Namespace: {NAMESPACE}; top_k: {top_k}")
    print(
        "Previously abstained: "
        + ", ".join(str(row["id"]) for row in previously_abstained)
    )
    print("Type-matched controls: " + ", ".join(str(row["id"]) for row in controls))

    made_call = False
    for row in selected_rows:
        row_id = str(row["id"])
        for attempt in range(1, ATTEMPTS_PER_ROW + 1):
            key = (row_id, attempt)
            if key in completed:
                continue
            if made_call:
                time.sleep(SLEEP_SECONDS)

            result = run_rag(row["question"], NAMESPACE, top_k)
            record = {
                "id": row_id,
                "attempt": attempt,
                "answer": result.get("answer"),
                "answered_by": result.get("answered_by"),
                "contexts": result.get("contexts") or [],
                "generation_ms": result.get("generation_ms"),
                "error": result.get("error"),
            }
            records_by_key[key] = record
            completed.add(key)
            _write_attempts(
                OUTPUT_PATH,
                [
                    records_by_key[record_key]
                    for record_key in sorted(records_by_key)
                ],
            )
            made_call = True

            if record["error"]:
                print(f"GENERATION ERROR {row_id} attempt {attempt}: {record['error']}")
            if record["answered_by"] != primary_generator:
                print(
                    f"NON-PRIMARY {row_id} attempt {attempt}: "
                    f"{record['answered_by']!r}"
                )

    expected = len(selected_rows) * ATTEMPTS_PER_ROW
    print(f"Saved {len(completed)}/{expected} attempts to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
