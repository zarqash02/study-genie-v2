"""Run and ingest the app's RAG path for evaluation."""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import time
from contextlib import redirect_stdout
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.config import config
from backend.services import embedding_service, llm_service, pdf_service, pinecone_service
from evaluation.config import PROJECT_ROOT, eval_config


class RetryableGenerationError(RuntimeError):
    pass


def _answered_by_from_logs(logs: str) -> Optional[str]:
    matches = re.findall(
        r"^\[LLM\] success via .+? / (.+?)\s*$",
        logs,
        flags=re.MULTILINE,
    )
    return matches[-1] if matches else None


def _require_eval_namespace(namespace: str) -> None:
    if not namespace.startswith(eval_config.pinecone_namespace_prefix):
        raise ValueError(
            "Evaluation namespace must start with "
            f"{eval_config.pinecone_namespace_prefix!r}"
        )


def _namespace_vector_count(index, namespace: str) -> int:
    stats = index.describe_index_stats()
    namespace_stats = getattr(stats, "namespaces", None)
    if namespace_stats is None and isinstance(stats, dict):
        namespace_stats = stats.get("namespaces")
    if namespace_stats is None:
        return 0
    summary = namespace_stats.get(namespace) if isinstance(namespace_stats, dict) else getattr(namespace_stats, namespace, None)
    if summary is None:
        return 0
    if isinstance(summary, dict):
        return int(summary.get("vector_count", 0))
    return int(getattr(summary, "vector_count", 0))


def _manifest_path(namespace: str) -> Path:
    _require_eval_namespace(namespace)
    return PROJECT_ROOT / "evaluation" / "cache" / f"{namespace}_chunks.json"


def ingest_eval_docs(namespace: str, reingest: bool = False) -> Dict:
    _require_eval_namespace(namespace)
    index = pinecone_service.get_index()
    existing_count = _namespace_vector_count(index, namespace)

    if existing_count > 0 and not reingest:
        manifest = _manifest_path(namespace)
        if not manifest.exists():
            raise RuntimeError(
                f"Namespace {namespace!r} contains vectors, but {manifest} does not exist."
            )
        return {
            "namespace": namespace,
            "ingested": False,
            "vector_count": existing_count,
            "manifest": str(manifest.relative_to(PROJECT_ROOT)),
        }

    if reingest:
        if existing_count > 0:
            index.delete(delete_all=True, namespace=namespace)

    docs_dir = PROJECT_ROOT / "evaluation" / "docs"
    pdf_paths = sorted(docs_dir.glob("*.pdf"), key=lambda path: path.name.lower())
    manifest_entries: List[Dict] = []

    for pdf_path in pdf_paths:
        digest = hashlib.sha256(pdf_path.name.encode("utf-8")).hexdigest()[:12]
        doc_id = f"eval-{digest}"
        records = pdf_service.process_pdf(
            file_bytes=pdf_path.read_bytes(),
            doc_id=doc_id,
            doc_name=pdf_path.name,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        if not records:
            continue

        vectors = embedding_service.embed_texts(
            [record["text"] for record in records]
        )
        pinecone_service.upsert_chunks(records, vectors, namespace=namespace)

        for record in records:
            chunk_index = int(record["id"].rsplit("_", 1)[1])
            manifest_entries.append(
                {
                    "doc": pdf_path.name,
                    "chunk_index": chunk_index,
                    "text": record["text"],
                }
            )

    manifest = _manifest_path(namespace)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(manifest_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "namespace": namespace,
        "ingested": True,
        "vector_count": _namespace_vector_count(index, namespace),
        "manifest": str(manifest.relative_to(PROJECT_ROOT)),
        "pdf_count": len(pdf_paths),
        "chunk_count": len(manifest_entries),
    }


def retrieve(query: str, namespace: str, top_k: int | None = None) -> List[Dict]:
    _require_eval_namespace(namespace)
    vector = embedding_service.embed_query(query)
    # Mirrors backend/main.py: embed query, then Pinecone query.
    return pinecone_service.query(
        vector,
        top_k=top_k or eval_config.top_k,
        namespace=namespace,
    )


def _is_retryable_generation_result(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(r"\b429\b|rate.?limit|too many requests", lower)
        or re.search(r"\b503\b|service unavailable", lower)
        or "timeout" in lower
        or "timed out" in lower
        or "deadline exceeded" in lower
    )


@retry(
    retry=retry_if_exception_type(RetryableGenerationError),
    stop=stop_after_attempt(eval_config.max_retries),
    wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
    reraise=True,
)
def _generate_answer_with_retry(
    query: str, matches: List[Dict]
) -> Tuple[str, Optional[str]]:
    captured_stdout = io.StringIO()
    original_stdout = sys.stdout
    try:
        with redirect_stdout(captured_stdout):
            answer_text = llm_service.answer_question(query, matches)
    finally:
        original_stdout.write(captured_stdout.getvalue())
        original_stdout.flush()
    answered_by = _answered_by_from_logs(captured_stdout.getvalue())
    if _is_retryable_generation_result(answer_text):
        raise RetryableGenerationError(answer_text)
    return answer_text, answered_by


def run_rag(
    question: str,
    namespace: str,
    top_k: int | None = None,
) -> Dict:
    started = time.perf_counter()
    retrieval_started = time.perf_counter()
    try:
        # Mirrors backend/main.py lines 89-90: retrieve first.
        matches = retrieve(question, namespace, top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        answered_by = None

        if not matches:
            answer_text = "I don't know based on the provided material."
            contexts: List[str] = []
            generation_ms = 0.0
        else:
            contexts = [
                match["metadata"].get("text", "")
                for match in matches
                if match["metadata"].get("text", "")
            ]
            generation_started = time.perf_counter()
            # Mirrors backend/main.py: build prompt and generate in
            # llm_service.answer_question after retrieval.
            answer_text, answered_by = _generate_answer_with_retry(question, matches)
            generation_ms = (time.perf_counter() - generation_started) * 1000.0

        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "answer": answer_text,
            "answered_by": answered_by,
            "contexts": contexts,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "error": None,
        }
    except Exception as exc:
        retrieval_ms = (
            (time.perf_counter() - retrieval_started) * 1000.0
            if "retrieval_ms" not in locals()
            else retrieval_ms
        )
        generation_ms = locals().get("generation_ms", 0.0)
        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "answer": None,
            "answered_by": None,
            "contexts": [],
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = ArgumentParser()
    parser.add_argument("--namespace", default="eval-phase2")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--reingest", action="store_true")
    parser.add_argument(
        "--question",
        default="What is an activation function?",
    )
    args = parser.parse_args()

    ingest_result = ingest_eval_docs(args.namespace, reingest=args.reingest)
    print(
        json.dumps(
            {"ingest": ingest_result},
            ensure_ascii=False,
        )
    )

    result = run_rag(args.question, args.namespace, args.top_k)
    print(json.dumps({
        "question": args.question,
        "answer": result["answer"],
        "context_count": len(result["contexts"]),
        "retrieval_ms": result["retrieval_ms"],
        "generation_ms": result["generation_ms"],
        "total_ms": result["total_ms"],
        "error": result["error"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
