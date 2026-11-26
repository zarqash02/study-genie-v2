"""Build a Markdown comparison from per-tag evaluation summaries."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
OUTPUT_PATH = RESULTS_DIR / "comparison.md"

METRICS = (
    (
        "faithfulness",
        "Faithfulness",
        ("aggregates", "ragas", "faithfulness", "mean"),
    ),
    (
        "answer_relevancy",
        "Answer relevancy",
        ("aggregates", "ragas", "response_relevancy", "mean"),
    ),
    (
        "context_precision",
        "Context precision",
        ("aggregates", "ragas", "context_precision", "mean"),
    ),
    (
        "context_recall",
        "Context recall",
        ("aggregates", "ragas", "context_recall", "mean"),
    ),
    (
        "hit_at_k",
        "Hit@k",
        ("aggregates", "retrieval", "single_hit_at_k", "mean"),
    ),
    (
        "mrr",
        "MRR",
        ("aggregates", "retrieval", "single_mrr", "mean"),
    ),
    (
        "abstention_rate",
        "Abstention rate",
        ("aggregates", "abstention", "rate"),
    ),
    (
        "p50_latency",
        "p50 latency (ms)",
        ("aggregates", "latency", "total_ms", "p50"),
    ),
)
CONFIG_COLUMNS = (
    ("chunk_size", "Chunk size"),
    ("overlap", "Overlap"),
    ("top_k", "Top-k"),
)


def _read_summaries() -> List[Dict]:
    summaries = []
    for path in sorted(RESULTS_DIR.glob("*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read evaluation summary {path}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
            raise ValueError(f"Invalid evaluation summary structure: {path}")
        data["_tag"] = str(data["config"].get("tag") or path.parent.name)
        summaries.append(data)
    if not summaries:
        raise FileNotFoundError(f"No evaluation summaries found under {RESULTS_DIR}.")
    return summaries


def _get_path(data: Dict, path: Tuple[str, ...]) -> Optional[float]:
    value = data
    try:
        for key in path:
            value = value[key]
    except (KeyError, TypeError):
        return None
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _format_metric(summary: Dict, key: str, path: Tuple[str, ...]) -> str:
    aggregate = _get_path(summary, path)
    if aggregate is None:
        return "n/a"

    parent = summary
    for component in path[:-1]:
        if not isinstance(parent, dict):
            parent = {}
            break
        parent = parent.get(component, {})
    n_valid = parent.get("n_valid") if isinstance(parent, dict) else None
    ci = parent.get("ci95") if isinstance(parent, dict) else None
    if isinstance(ci, list) and len(ci) == 2 and all(value is not None for value in ci):
        value_text = f"{aggregate!r} [{float(ci[0])!r}, {float(ci[1])!r}]"
    else:
        value_text = repr(aggregate)
    if n_valid is not None:
        value_text += f" (n={n_valid})"
    return value_text


def _best_values(summaries: List[Dict]) -> Dict[str, List[float]]:
    best = {}
    for key, _, path in METRICS:
        values = [
            value
            for summary in summaries
            if (value := _get_path(summary, path)) is not None
        ]
        if values:
            best[key] = [min(values) if key == "p50_latency" else max(values)]
    return best


def build_comparison(summaries: List[Dict]) -> str:
    best = _best_values(summaries)
    headers = [
        "Tag",
        *(label for _, label, _ in METRICS),
        *(label for _, label in CONFIG_COLUMNS),
    ]
    lines = [
        "# Evaluation comparison",
        "",
        "Bold marks the best observed value in each metric column "
        "(higher is better, except p50 latency where lower is better). "
        "Chunk settings are descriptive and are not ranked.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for summary in summaries:
        cells = [summary["_tag"]]
        for key, _, path in METRICS:
            formatted = _format_metric(summary, key, path)
            value = _get_path(summary, path)
            if value is not None and value in best.get(key, []):
                formatted = f"**{formatted}**"
            cells.append(formatted)
        config = summary["config"]
        cells.extend(
            str(config.get(key)) if config.get(key) is not None else "n/a"
            for key, _ in CONFIG_COLUMNS
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    summaries = _read_summaries()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_comparison(summaries), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(summaries)} tags).")


if __name__ == "__main__":
    main()
