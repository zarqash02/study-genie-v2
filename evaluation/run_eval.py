"""Run resumable retrieval, generation, and RAGAS evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from backend.config import config as app_config
from backend.services import llm_service, pdf_service, pinecone_service
from evaluation.config import PROJECT_ROOT, eval_config
from evaluation.rag_runner import ingest_eval_docs, run_rag


REFUSAL_TEXT = "I don't know based on the provided material."
METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "response_relevancy": "answer_relevancy",
    "context_precision": "llm_context_precision_with_reference",
    "context_recall": "context_recall",
}
JUDGE_MAX_OUTPUT_TOKENS = 8192
PREVIOUS_JUDGE_MAX_OUTPUT_TOKENS = (900, 2048)
PER_ROW_FIELDS = [
    "id",
    "type",
    "question",
    "answer",
    "reference",
    "contexts",
    *METRIC_COLUMNS,
    "hit",
    "rr",
    "gold_coverage",
    "retrieval_ms",
    "generation_ms",
    "total_ms",
    "error",
]


def validate_testset(path: Path, allow_unreviewed: bool = False) -> Tuple[List[Dict], bool]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Testset must be a non-empty JSON list.")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every testset row must be a JSON object.")

    is_draft = path.name.casefold() == "testset_draft.json"
    has_unreviewed = any(row.get("reviewed") is not True for row in rows)
    unreviewed = is_draft or has_unreviewed
    if unreviewed and not allow_unreviewed:
        reasons = []
        if is_draft:
            reasons.append("testset_draft.json is a draft")
        if has_unreviewed:
            reasons.append("one or more rows have reviewed=false")
        raise ValueError(
            "Refusing to evaluate an unreviewed test set: "
            + "; ".join(reasons)
            + ". Review every row, set reviewed=true, save as evaluation/testset.json, "
            "or explicitly pass --allow-unreviewed."
        )

    required = {"id", "type", "question", "reference", "gold_chunks", "source_docs"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Testset row is missing required fields: {sorted(missing)}")
        if not isinstance(row["gold_chunks"], list):
            raise ValueError(f"gold_chunks must be a list for row {row['id']!r}.")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Testset row ids must be unique.")
    return rows, unreviewed


def _validate_tag(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag) or tag in {".", ".."}:
        raise ValueError("Tag must contain only letters, digits, dot, underscore, or hyphen.")
    return tag


def _validate_namespace(namespace: str) -> str:
    if not namespace.startswith(eval_config.pinecone_namespace_prefix):
        raise ValueError(
            "Evaluation namespace must start with "
            f"{eval_config.pinecone_namespace_prefix!r}; refusing vector operation."
        )
    return namespace


def _delete_eval_namespace(namespace: str) -> None:
    _validate_namespace(namespace)
    pinecone_service.get_index().delete(delete_all=True, namespace=namespace)


def _fingerprint(row: Dict) -> str:
    payload = {
        key: row.get(key)
        for key in ("type", "question", "reference", "gold_chunks")
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _score_fingerprint(row: Dict, max_output_tokens: Optional[int] = None) -> str:
    payload = {
        "row": _fingerprint(row),
        "judge_model": eval_config.judge_model,
        "judge_max_output_tokens": max_output_tokens,
        "embedding_model": eval_config.embedding_model,
        "ragas_version": version("ragas"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _score_cache_matches(record: Dict, row: Dict) -> bool:
    fingerprint = record.get("input_hash")
    compatible_fingerprints = {
        _score_fingerprint(row, JUDGE_MAX_OUTPUT_TOKENS)
    }
    compatible_fingerprints.update(
        _score_fingerprint(row, previous_max_tokens)
        for previous_max_tokens in PREVIOUS_JUDGE_MAX_OUTPUT_TOKENS
    )
    return fingerprint in compatible_fingerprints


def _load_jsonl(path: Path) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"Invalid output cache record at {path}:{line_number}.")
            records[str(record["id"])] = record
    return records


def _write_jsonl_atomic(path: Path, records: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records.values():
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    temp_path.replace(path)


def _generation_outputs(
    rows: Sequence[Dict],
    namespace: str,
    top_k: int,
    cache_path: Path,
    skip_generation: bool,
    sleep_seconds: float,
) -> Dict[str, Dict]:
    outputs = _load_jsonl(cache_path)
    calls_to_run = [
        row
        for row in rows
        if not (
            (cached := outputs.get(str(row["id"])))
            and cached.get("input_hash") == _fingerprint(row)
            and not cached.get("error")
        )
    ]

    with tqdm(total=len(rows), desc="RAG generation", unit="row") as progress:
        for row in rows:
            row_id = str(row["id"])
            cached = outputs.get(row_id)
            if (
                cached
                and cached.get("input_hash") == _fingerprint(row)
                and not cached.get("error")
            ):
                progress.update(1)
                continue

            if skip_generation:
                outputs[row_id] = {
                    "id": row_id,
                    "input_hash": _fingerprint(row),
                    "answer": None,
                    "answered_by": None,
                    "contexts": [],
                    "retrieval_ms": None,
                    "generation_ms": None,
                    "total_ms": None,
                    "error": "No successful cached output; generation skipped.",
                }
                progress.update(1)
                continue

            result = run_rag(row["question"], namespace, top_k)
            answer = result.get("answer")
            error = result.get("error")
            if isinstance(answer, str) and answer.startswith("[All providers failed."):
                error = error or "The app's LLM provider chain failed."
            outputs[row_id] = {
                "id": row_id,
                "input_hash": _fingerprint(row),
                "answer": answer,
                "answered_by": result.get("answered_by"),
                "contexts": result.get("contexts") or [],
                "retrieval_ms": result.get("retrieval_ms"),
                "generation_ms": result.get("generation_ms"),
                "total_ms": result.get("total_ms"),
                "error": error,
            }
            _write_jsonl_atomic(cache_path, outputs)
            progress.update(1)

            if sleep_seconds > 0 and row is not calls_to_run[-1]:
                time.sleep(sleep_seconds)

    return {str(row["id"]): outputs[str(row["id"])] for row in rows}


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _retrieval_metrics(row: Dict, output: Dict) -> Dict[str, Optional[float]]:
    retrieved = output.get("contexts") or []
    retrieved_norm = [_normalized_text(str(context)) for context in retrieved]
    gold_chunks = row.get("gold_chunks") or []

    if row["type"] == "single":
        gold = _normalized_text(gold_chunks[0])
        rank = next(
            (index for index, context in enumerate(retrieved_norm, start=1) if context == gold),
            None,
        )
        return {
            "hit": 1.0 if rank is not None else 0.0,
            "rr": 1.0 / rank if rank is not None else 0.0,
            "gold_coverage": None,
        }
    if row["type"] == "multi":
        gold_norm = {_normalized_text(chunk) for chunk in gold_chunks}
        found = sum(1 for chunk in gold_norm if chunk in set(retrieved_norm))
        return {
            "hit": None,
            "rr": None,
            "gold_coverage": found / len(gold_norm) if gold_norm else 0.0,
        }
    return {"hit": None, "rr": None, "gold_coverage": None}


def _contains_refusal(answer: Optional[str]) -> bool:
    return isinstance(answer, str) and REFUSAL_TEXT.casefold() in answer.casefold()


def _load_scores(path: Path) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"id", "input_hash"}.issubset(reader.fieldnames):
            raise ValueError(f"Score cache has an invalid header: {path}")
        for row in reader:
            records[row["id"]] = row
    return records


def _metric_components(max_workers: int, metric_names: Sequence[str]):
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env or evaluation/.env.")
    judge = LangchainLLMWrapper(
        ChatGroq(
            model=eval_config.judge_model,
            temperature=eval_config.judge_temperature,
            max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            api_key=api_key,
        )
    )
    embedding_model = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model=eval_config.embedding_model)
    )
    available_metrics = {
        "faithfulness": Faithfulness(),
        "response_relevancy": ResponseRelevancy(strictness=1),
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }
    metrics = [available_metrics[name] for name in metric_names]
    run_config = RunConfig(
        timeout=180,
        max_retries=6,
        max_wait=60,
        max_workers=max_workers,
        seed=eval_config.random_seed,
    )
    return evaluate, EvaluationDataset, metrics, judge, embedding_model, run_config


def _append_score_batch(
    path: Path,
    rows: Sequence[Dict],
    score_frame,
    existing_scores: Dict[str, Dict],
    metric_names: Sequence[str],
) -> Dict[str, Dict]:
    columns = [METRIC_COLUMNS[name] for name in metric_names]
    missing_columns = [column for column in columns if column not in score_frame.columns]
    if missing_columns:
        raise RuntimeError(
            "RAGAS result omitted expected metric columns: "
            + ", ".join(missing_columns)
        )
    if len(score_frame) != len(rows):
        raise RuntimeError(
            f"RAGAS returned {len(score_frame)} scores for a batch of {len(rows)} rows."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    batch_records = {}
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "input_hash", *METRIC_COLUMNS],
        )
        if write_header:
            writer.writeheader()
        for position, row in enumerate(rows):
            values = {}
            for name, result_column in METRIC_COLUMNS.items():
                previous = existing_scores.get(str(row["id"]), {})
                if name in metric_names:
                    value = score_frame.iloc[position][result_column]
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        numeric = math.nan
                else:
                    try:
                        numeric = float(previous.get(name, ""))
                    except (TypeError, ValueError):
                        numeric = math.nan
                if math.isfinite(numeric):
                    values[name] = numeric
                else:
                    previous_value = previous.get(name, "")
                    try:
                        previous_numeric = float(previous_value)
                    except (TypeError, ValueError):
                        previous_numeric = math.nan
                    values[name] = (
                        previous_numeric
                        if _score_cache_matches(previous, row)
                        and math.isfinite(previous_numeric)
                        else ""
                    )
            record = {
                "id": str(row["id"]),
                "input_hash": _score_fingerprint(row, JUDGE_MAX_OUTPUT_TOKENS),
                **values,
            }
            writer.writerow(record)
            batch_records[record["id"]] = record
        file.flush()
        os.fsync(file.fileno())
    return batch_records


def _score_rows(
    rows: Sequence[Dict],
    outputs: Dict[str, Dict],
    score_path: Path,
    skip_scoring: bool,
    max_workers: int,
) -> Dict[str, Dict]:
    scores = _load_scores(score_path)
    eligible = [
        row
        for row in rows
        if row["type"] != "unanswerable"
        and not outputs[str(row["id"])].get("error")
        and isinstance(outputs[str(row["id"])].get("answer"), str)
        and bool(outputs[str(row["id"])]["answer"].strip())
    ]
    if skip_scoring or not eligible:
        return scores

    todo_by_metrics: Dict[Tuple[str, ...], List[Dict]] = defaultdict(list)
    for row in eligible:
        missing_metrics = tuple(
            name
            for name in METRIC_COLUMNS
            if _score_value(row, scores, name) is None
        )
        if missing_metrics:
            todo_by_metrics[missing_metrics].append(row)
    if not todo_by_metrics:
        return scores

    for metric_names, todo in todo_by_metrics.items():
        evaluate, EvaluationDataset, metrics, judge, embeddings, run_config = (
            _metric_components(max_workers, metric_names)
        )
        for offset in tqdm(
            range(0, len(todo), 10),
            desc=f"RAGAS scoring ({','.join(metric_names)})",
            unit="batch",
        ):
            batch = todo[offset : offset + 10]
            samples = [
                {
                    "user_input": row["question"],
                    "response": outputs[str(row["id"])]["answer"],
                    "retrieved_contexts": outputs[str(row["id"])]["contexts"],
                    "reference": row["reference"],
                }
                for row in batch
            ]
            dataset = EvaluationDataset.from_list(samples)
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=judge,
                embeddings=embeddings,
                run_config=run_config,
                show_progress=False,
                raise_exceptions=False,
            )
            score_frame = result.to_pandas()
            scores.update(
                _append_score_batch(
                    score_path,
                    batch,
                    score_frame,
                    scores,
                    metric_names,
                )
            )
    return scores


def _score_value(
    row: Dict,
    scores: Dict[str, Dict],
    metric: str,
) -> Optional[float]:
    record = scores.get(str(row["id"]))
    if not record or not _score_cache_matches(record, row):
        return None
    value = record.get(metric)
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _bootstrap(values: Iterable[Optional[float]], rng: np.random.Generator) -> Dict:
    valid = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=float,
    )
    if not len(valid):
        return {"mean": None, "ci95": [None, None], "n_valid": 0}
    means = rng.choice(valid, size=(1000, len(valid)), replace=True).mean(axis=1)
    return {
        "mean": float(valid.mean()),
        "ci95": [float(x) for x in np.percentile(means, [2.5, 97.5])],
        "n_valid": int(len(valid)),
    }


def _percentiles(values: Iterable[Optional[float]]) -> Dict:
    valid = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=float,
    )
    if not len(valid):
        return {"p50": None, "p95": None, "n": 0}
    p50, p95 = np.percentile(valid, [50, 95])
    return {"p50": float(p50), "p95": float(p95), "n": int(len(valid))}


def _docs_summary(manifest: Sequence[Dict]) -> List[Dict]:
    docs_dir = PROJECT_ROOT / "evaluation" / "docs"
    chunks_per_doc: Dict[str, int] = defaultdict(int)
    for chunk in manifest:
        chunks_per_doc[chunk["doc"]] += 1

    result = []
    for path in sorted(docs_dir.glob("*.pdf"), key=lambda item: item.name.casefold()):
        pages = pdf_service.extract_pdf_pages(path.read_bytes())
        result.append(
            {
                "doc": path.name,
                "page_count": len(pages),
                "chunk_count": chunks_per_doc.get(path.name, 0),
            }
        )
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _aggregate_results(
    rows: Sequence[Dict],
    outputs: Dict[str, Dict],
    scores: Dict[str, Dict],
    max_top_k: int,
) -> Tuple[Dict, Dict[str, Dict]]:
    rng = np.random.default_rng(eval_config.random_seed)
    per_row_metrics = {}
    for row in rows:
        output = outputs[str(row["id"])]
        per_row_metrics[str(row["id"])] = _retrieval_metrics(row, output)

    aggregates = {"ragas": {}, "retrieval": {}, "abstention": {}, "latency": {}}
    nan_metrics = {}
    for metric in METRIC_COLUMNS:
        values = [_score_value(row, scores, metric) for row in rows]
        scored_ids = [
            str(row["id"])
            for row in rows
            if _score_cache_matches(scores.get(str(row["id"]), {}), row)
        ]
        affected = [
            row_id
            for row_id, value in zip(
                [str(row["id"]) for row in rows],
                [_score_value(row, scores, metric) for row in rows],
            )
            if row_id in scored_ids and value is None
        ]
        nan_metrics[metric] = {"count": len(affected), "ids": affected}
        aggregates["ragas"][metric] = _bootstrap(values, rng)

    by_type = {}
    for question_type in sorted({row["type"] for row in rows}):
        type_rows = [row for row in rows if row["type"] == question_type]
        type_aggregates = {"ragas": {}, "retrieval": {}}
        for metric in METRIC_COLUMNS:
            type_aggregates["ragas"][metric] = _bootstrap(
                [_score_value(row, scores, metric) for row in type_rows], rng
            )
        if question_type == "single":
            type_aggregates["retrieval"] = {
                "hit_at_k": _bootstrap(
                    [per_row_metrics[str(row["id"])]["hit"] for row in type_rows],
                    rng,
                ),
                "mrr": _bootstrap(
                    [per_row_metrics[str(row["id"])]["rr"] for row in type_rows],
                    rng,
                ),
            }
        elif question_type == "multi":
            type_aggregates["retrieval"] = {
                "gold_coverage": _bootstrap(
                    [
                        per_row_metrics[str(row["id"])]["gold_coverage"]
                        for row in type_rows
                    ],
                    rng,
                )
            }
        by_type[question_type] = type_aggregates

    single_rows = [row for row in rows if row["type"] == "single"]
    multi_rows = [row for row in rows if row["type"] == "multi"]
    aggregates["retrieval"] = {
        "k": max_top_k,
        "single_hit_at_k": _bootstrap(
            [per_row_metrics[str(row["id"])]["hit"] for row in single_rows], rng
        ),
        "single_mrr": _bootstrap(
            [per_row_metrics[str(row["id"])]["rr"] for row in single_rows], rng
        ),
        "multi_gold_coverage": _bootstrap(
            [
                per_row_metrics[str(row["id"])]["gold_coverage"]
                for row in multi_rows
            ],
            rng,
        ),
    }

    unanswerable_rows = [row for row in rows if row["type"] == "unanswerable"]
    answerable_rows = [row for row in rows if row["type"] != "unanswerable"]
    abstained_unanswerable = sum(
        _contains_refusal(outputs[str(row["id"])].get("answer"))
        for row in unanswerable_rows
    )
    false_abstentions = sum(
        _contains_refusal(outputs[str(row["id"])].get("answer"))
        for row in answerable_rows
    )
    aggregates["abstention"] = {
        "unanswerable_abstained": int(abstained_unanswerable),
        "unanswerable_total": len(unanswerable_rows),
        "rate": (
            abstained_unanswerable / len(unanswerable_rows)
            if unanswerable_rows
            else None
        ),
        "answerable_false_abstentions": int(false_abstentions),
        "answerable_total": len(answerable_rows),
        "false_abstention_rate": (
            false_abstentions / len(answerable_rows) if answerable_rows else None
        ),
    }

    aggregates["latency"] = {
        name: _percentiles(
            [outputs[str(row["id"])].get(name) for row in rows]
        )
        for name in ("retrieval_ms", "generation_ms", "total_ms")
    }
    aggregates["nan_metrics"] = nan_metrics
    aggregates["by_type"] = by_type
    return aggregates, per_row_metrics


def _write_per_row(
    path: Path,
    rows: Sequence[Dict],
    outputs: Dict[str, Dict],
    scores: Dict[str, Dict],
    retrieval_metrics: Dict[str, Dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PER_ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            row_id = str(row["id"])
            output = outputs[row_id]
            retrieval = retrieval_metrics[row_id]
            record = {
                "id": row_id,
                "type": row["type"],
                "question": row["question"],
                "answer": output.get("answer") or "",
                "reference": row["reference"],
                "contexts": " ||| ".join(output.get("contexts") or []),
                **{
                    metric: _score_value(row, scores, metric)
                    for metric in METRIC_COLUMNS
                },
                **retrieval,
                "retrieval_ms": output.get("retrieval_ms"),
                "generation_ms": output.get("generation_ms"),
                "total_ms": output.get("total_ms"),
                "error": output.get("error") or "",
            }
            writer.writerow(record)


def _format_mean_ci(aggregate: Dict) -> str:
    mean = aggregate.get("mean")
    ci = aggregate.get("ci95")
    if mean is None or not ci or ci[0] is None:
        return f"n/a (n={aggregate.get('n_valid', 0)})"
    return f"{mean!r} [{ci[0]!r}, {ci[1]!r}] (n={aggregate['n_valid']})"


def _write_summary_markdown(path: Path, summary: Dict) -> None:
    config = summary["config"]
    aggregates = summary["aggregates"]
    lines = [
        f"# Evaluation: {config['tag']}",
        "",
        f"- Date: {config['date']}",
        f"- Namespace: `{config['namespace']}`",
        f"- Test rows: {config['n_total']}; RAGAS-scored: {config['n_scored']}; "
        f"excluded: {config['n_excluded_from_scoring']}; errors: {config['n_errors']}",
        f"- Unanswerable rows: {config['n_unanswerable']}; "
        f"reviewed: {str(config['reviewed']).lower()}",
    ]
    if summary.get("warnings"):
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {warning}" for warning in summary["warnings"])

    lines += [
        "",
        "## RAGAS scores",
        "",
        "| Metric | Mean [95% bootstrap CI] | NaNs | Affected IDs |",
        "|---|---:|---:|---|",
    ]
    labels = {
        "faithfulness": "Faithfulness",
        "response_relevancy": "ResponseRelevancy",
        "context_precision": "LLMContextPrecisionWithReference",
        "context_recall": "LLMContextRecall",
    }
    for metric, aggregate in aggregates["ragas"].items():
        nan_info = aggregates["nan_metrics"][metric]
        lines.append(
            f"| {labels[metric]} | {_format_mean_ci(aggregate)} | "
            f"{nan_info['count']} | {', '.join(nan_info['ids']) or '—'} |"
        )

    retrieval = aggregates["retrieval"]
    lines += [
        "",
        "## Retrieval",
        "",
        f"Single rows (k={retrieval['k']}): hit@k "
        f"{_format_mean_ci(retrieval['single_hit_at_k'])}; MRR "
        f"{_format_mean_ci(retrieval['single_mrr'])}.",
        "",
        f"Multi rows: gold coverage "
        f"{_format_mean_ci(retrieval['multi_gold_coverage'])}.",
        "",
        "## Abstention",
        "",
        "| Measure | Count | Rate |",
        "|---|---:|---:|",
    ]
    abstention = aggregates["abstention"]
    rate = abstention["rate"]
    false_rate = abstention["false_abstention_rate"]
    abstention_rate = repr(rate) if rate is not None else "n/a"
    lines.append(
        f"| Unanswerable abstained | {abstention['unanswerable_abstained']}/"
        f"{abstention['unanswerable_total']} | {abstention_rate} |"
    )
    lines.append(
        f"| False abstentions on answerable | "
        f"{abstention['answerable_false_abstentions']}/"
        f"{abstention['answerable_total']} | "
        f"{false_rate!r} |" if false_rate is not None else
        f"| False abstentions on answerable | "
        f"{abstention['answerable_false_abstentions']}/"
        f"{abstention['answerable_total']} | n/a |"
    )

    lines += [
        "",
        "## Latency",
        "",
        "| Stage | p50 ms | p95 ms | n |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in aggregates["latency"].items():
        lines.append(
            f"| {name} | {stats['p50']!r} | {stats['p95']!r} | {stats['n']} |"
            if stats["p50"] is not None
            else f"| {name} | n/a | n/a | {stats['n']} |"
        )

    lines += [
        "",
        "## By question type",
        "",
        "| Type | Faithfulness | Response relevancy | Context precision | Context recall | Retrieval |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for question_type, stats in aggregates["by_type"].items():
        retrieval_values = stats["retrieval"]
        if question_type == "single":
            retrieval_text = (
                f"hit@k {_format_mean_ci(retrieval_values['hit_at_k'])}; "
                f"MRR {_format_mean_ci(retrieval_values['mrr'])}"
            )
        elif question_type == "multi":
            retrieval_text = _format_mean_ci(retrieval_values["gold_coverage"])
        else:
            retrieval_text = "—"
        lines.append(
            f"| {question_type} | "
            + " | ".join(
                _format_mean_ci(stats["ragas"][metric])
                for metric in METRIC_COLUMNS
            )
            + f" | {retrieval_text} |"
        )

    lines += [
        "",
        "## Source documents",
        "",
        "| Document | Pages | Chunks |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {doc['doc']} | {doc['page_count']} | {doc['chunk_count']} |"
        for doc in config["docs"]
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_worst_rows(
    path: Path,
    rows: Sequence[Dict],
    outputs: Dict[str, Dict],
    scores: Dict[str, Dict],
) -> None:
    lines = ["# Lowest-scoring rows", ""]
    for metric, heading in (
        ("faithfulness", "Lowest Faithfulness"),
        ("context_recall", "Lowest Context Recall"),
    ):
        scored = [
            (value, row)
            for row in rows
            if (value := _score_value(row, scores, metric)) is not None
        ]
        scored.sort(key=lambda pair: pair[0])
        lines += [f"## {heading}", ""]
        if not scored:
            lines += ["No valid scores for this metric.", ""]
            continue
        for value, row in scored[:10]:
            output = outputs[str(row["id"])]
            lines += [
                f"### {row['id']} — {metric}: {value!r}",
                "",
                f"- Question: {row['question']}",
                f"- Answer: {output.get('answer') or ''}",
                f"- Reference: {row['reference']}",
                "- Contexts:",
            ]
            contexts = output.get("contexts") or []
            if contexts:
                lines.extend(
                    f"  - {context[:300]}" for context in contexts
                )
            else:
                lines.append("  - (none)")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_summary(
    tag: str,
    namespace: str,
    top_k: int,
    rows: Sequence[Dict],
    unreviewed: bool,
    manifest: Sequence[Dict],
    outputs: Dict[str, Dict],
    scores: Dict[str, Dict],
    aggregates: Dict,
) -> Dict:
    eligible = [
        row
        for row in rows
        if row["type"] != "unanswerable"
        and not outputs[str(row["id"])].get("error")
        and isinstance(outputs[str(row["id"])].get("answer"), str)
        and bool(outputs[str(row["id"])]["answer"].strip())
    ]
    n_score_attempted = sum(
        1
        for row in eligible
        if _score_cache_matches(scores.get(str(row["id"]), {}), row)
    )
    n_scored = sum(
        1
        for row in eligible
        if any(
            _score_value(row, scores, metric) is not None
            for metric in METRIC_COLUMNS
        )
    )
    n_answerable = sum(row["type"] != "unanswerable" for row in rows)
    generator_model = app_config.GROQ_MODEL
    fallback_models = [model for _, model in llm_service.HF_CHAIN]
    warnings = []
    if eval_config.judge_model.casefold() == generator_model.casefold():
        warnings.append(
            "Judge model is the same as the configured generator model; "
            "evaluation may be self-biased."
        )
    elif eval_config.judge_model.casefold() in {
        model.casefold() for model in fallback_models
    }:
        warnings.append(
            "Judge model appears in the configured generator fallback chain; "
            "evaluation may be self-biased."
        )
    return {
        "config": {
            "tag": tag,
            "date": datetime.now(timezone.utc).isoformat(),
            "ragas_version": version("ragas"),
            "judge_model": eval_config.judge_model,
            "generator_model_id": generator_model,
            "embedding_model": eval_config.embedding_model,
            "chunk_size": app_config.CHUNK_SIZE,
            "overlap": app_config.CHUNK_OVERLAP,
            "top_k": top_k,
            "namespace": namespace,
            "n_total": len(rows),
            "n_scored": n_scored,
            "n_score_attempted": n_score_attempted,
            "n_eligible_for_scoring": len(eligible),
            "n_excluded_from_scoring": len(rows) - len(eligible),
            "n_errors": sum(bool(outputs[str(row["id"])].get("error")) for row in rows),
            "n_unanswerable": sum(row["type"] == "unanswerable" for row in rows),
            "reviewed": not unreviewed,
            "docs": _docs_summary(manifest),
        },
        "aggregates": aggregates,
        "warnings": warnings,
    }


def _write_reports(
    result_dir: Path,
    summary: Dict,
    rows: Sequence[Dict],
    outputs: Dict[str, Dict],
    scores: Dict[str, Dict],
    retrieval_metrics: Dict[str, Dict],
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_per_row(
        result_dir / "per_row.csv", rows, outputs, scores, retrieval_metrics
    )
    _write_worst_rows(result_dir / "worst_rows.md", rows, outputs, scores)
    _write_summary_markdown(result_dir / "summary.md", summary)
    (result_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--testset",
        type=Path,
        default=Path("evaluation") / "testset.json",
    )
    parser.add_argument("--reingest", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    return parser, parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser, args = _parse_args()

    try:
        tag = _validate_tag(args.tag)
        namespace = _validate_namespace(args.namespace or f"eval-{tag}")
        if args.top_k is not None and args.top_k <= 0:
            raise ValueError("--top-k must be positive.")
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive.")
        if args.max_workers <= 0:
            raise ValueError("--max-workers must be positive.")
        if args.sleep_seconds < 0:
            raise ValueError("--sleep-seconds cannot be negative.")
        rows, unreviewed = validate_testset(args.testset, args.allow_unreviewed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.limit is not None:
        rows = rows[: args.limit]
    top_k = args.top_k or eval_config.top_k
    cache_dir = PROJECT_ROOT / "evaluation" / "cache" / tag
    result_dir = PROJECT_ROOT / "evaluation" / "results" / tag
    output_cache = cache_dir / "outputs.jsonl"
    score_cache = cache_dir / "scores.csv"
    namespace_checked = _validate_namespace(namespace)

    try:
        ingest_result = ingest_eval_docs(namespace_checked, reingest=args.reingest)
        print(
            f"Ingestion: namespace={namespace_checked}, "
            f"vectors={ingest_result['vector_count']}, "
            f"ingested={ingest_result['ingested']}"
        )

        manifest_path = PROJECT_ROOT / ingest_result["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = _generation_outputs(
            rows,
            namespace_checked,
            top_k,
            output_cache,
            args.skip_generation,
            args.sleep_seconds,
        )
        scores = _score_rows(
            rows,
            outputs,
            score_cache,
            args.skip_scoring,
            args.max_workers,
        )
        aggregates, retrieval_metrics = _aggregate_results(
            rows, outputs, scores, top_k
        )
        summary = _build_summary(
            tag,
            namespace_checked,
            top_k,
            rows,
            unreviewed,
            manifest,
            outputs,
            scores,
            aggregates,
        )
        _write_reports(
            result_dir,
            summary,
            rows,
            outputs,
            scores,
            retrieval_metrics,
        )

        for warning in summary["warnings"]:
            print(f"WARNING: {warning}")
        if unreviewed:
            print("UNREVIEWED TEST SET")
        print(
            f"Results written to {result_dir.relative_to(PROJECT_ROOT)}; "
            f"n_total={summary['config']['n_total']}, "
            f"n_scored={summary['config']['n_scored']}, "
            f"n_errors={summary['config']['n_errors']}"
        )
    finally:
        if args.cleanup:
            _delete_eval_namespace(namespace_checked)
            print(f"Deleted vectors from evaluation namespace {namespace_checked}.")


if __name__ == "__main__":
    main()
