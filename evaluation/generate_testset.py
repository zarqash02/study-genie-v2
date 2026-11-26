"""Generate a human-reviewable testset draft from the evaluation chunk manifest."""

from __future__ import annotations

import json
import os
import random
import re
from argparse import ArgumentParser
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from groq import Groq
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from backend.services.pdf_service import extract_pdf_pages
from evaluation.config import PROJECT_ROOT, eval_config


DOCS_DIR = PROJECT_ROOT / "evaluation" / "docs"
DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "cache" / "eval-phase2_chunks.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "testset_draft.json"
QUESTIONS_PER_DOC = 5


class InvalidGeneration(RuntimeError):
    pass


def _is_transient_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError)) or any(
        word in type(exc).__name__.lower()
        for word in ("timeout", "connection", "ratelimit", "internalserver")
    )


@retry(
    retry=retry_if_exception(_is_transient_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=20, jitter=1),
    reraise=True,
)
def _completion(client: Groq, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Create evidence-grounded evaluation questions. Follow the "
                    "requested JSON schema exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise InvalidGeneration("The judge returned an empty response.")
    return content


def _parse_json_object(raw: str) -> Dict[str, str]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise InvalidGeneration("The judge response did not contain a JSON object.")


def _normalize_question(question: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", question.casefold()))


def _validate_question(question: object, seen_questions: set[str]) -> str:
    if not isinstance(question, str):
        raise InvalidGeneration("The question must be a string.")
    question = " ".join(question.split())
    if len(question.split()) < 6:
        raise InvalidGeneration("The question has fewer than six words.")
    if re.search(
        r"\b(?:the\s+|this\s+|that\s+|these\s+|those\s+|given\s+|provided\s+)?"
        r"(?:text|passage|chunk|context)\b",
        question,
        flags=re.IGNORECASE,
    ):
        raise InvalidGeneration("The question refers to source text or context.")
    normalized = _normalize_question(question)
    if not normalized or normalized in seen_questions:
        raise InvalidGeneration("The question duplicates an existing question.")
    return question


def _build_prompt(
    question_type: str,
    chunks: Sequence[Dict],
    prior_questions: Iterable[str],
) -> str:
    source = "\n\n".join(
        f"Source {index}:\n{chunk['text']}" for index, chunk in enumerate(chunks, 1)
    )
    prior = "\n".join(f"- {question}" for question in prior_questions)
    common = (
        "Return exactly one strict JSON object with string keys and values "
        '{"question": "...", "reference": "..."}. No markdown or extra text. '
        "The question must be standalone and contain at least six words. "
        "Never refer to 'the text', 'the passage', 'the chunk', or 'the context', "
        "and never say 'according to' any source. Do not duplicate or paraphrase "
        "any listed question. "
    )

    if question_type == "single":
        task = (
            "Write one factual question answerable using ONLY the single source "
            "below. The reference must answer it using ONLY facts stated in that "
            "source."
        )
    elif question_type == "multi":
        task = (
            "Write one question that genuinely requires combining information "
            "from BOTH same-document sources below. The reference must use ONLY "
            "facts from those sources and must need both to answer completely."
        )
    elif question_type == "summary":
        task = (
            "Write one conceptual question about the shared idea or progression "
            "across the three adjacent sources below. The reference must synthesize "
            "ONLY information stated in those sources."
        )
    elif question_type == "unanswerable":
        task = (
            "Write one plausible, standalone question about the same topic as the "
            "source, but whose answer is NOT stated or inferable from it. Return "
            'the exact reference string "NOT_ANSWERABLE".'
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    return (
        f"{common}{task}\n\n"
        f"Previous questions to avoid:\n{prior or '- (none)'}\n\n"
        f"Sources:\n{source}\n\n"
        'Return JSON with exactly the keys "question" and "reference".'
    )


def _generate_row(
    client: Groq,
    question_type: str,
    chunks: Sequence[Dict],
    seen_questions: set[str],
    question_history: Sequence[str],
    row_id: str,
) -> Dict:
    feedback = ""
    for _ in range(eval_config.max_retries):
        prompt = _build_prompt(question_type, chunks, question_history)
        if feedback:
            prompt += (
                "\nYour previous response was rejected because "
                f"{feedback} Generate a new response that fixes this."
            )
        raw = _completion(client, eval_config.judge_model, prompt)
        try:
            generated = _parse_json_object(raw)
            question = _validate_question(
                generated.get("question"), seen_questions
            )
        except InvalidGeneration as exc:
            feedback = str(exc)
            continue
        reference = generated.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            feedback = "the reference was missing or empty."
            continue

        if question_type == "unanswerable":
            if reference.strip() != "NOT_ANSWERABLE":
                feedback = 'the reference must be exactly "NOT_ANSWERABLE".'
                continue
            gold_chunks: List[str] = []
            reference = "NOT_ANSWERABLE"
        else:
            reference = reference.strip()
            if reference == "NOT_ANSWERABLE":
                feedback = "this question must be answerable from its source chunks."
                continue
            gold_chunks = [chunk["text"] for chunk in chunks]

        seen_questions.add(_normalize_question(question))
        question_history.append(question)
        return {
            "id": row_id,
            "type": question_type,
            "question": question,
            "reference": reference,
            "gold_chunks": gold_chunks,
            "source_docs": list(dict.fromkeys(chunk["doc"] for chunk in chunks)),
            "reviewed": False,
        }

    raise RuntimeError(
        f"Could not generate a valid, non-duplicate {question_type} question for {row_id} "
        f"after {eval_config.max_retries} attempts; last rejection: {feedback}"
    )


def _check_docs() -> List[Path]:
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"), key=lambda path: path.name.casefold())
    if not 3 <= len(pdf_paths) <= 5:
        raise RuntimeError(
            f"Expected 3-5 PDFs in {DOCS_DIR.relative_to(PROJECT_ROOT)}, found {len(pdf_paths)}."
        )

    text_pdfs = []
    for path in pdf_paths:
        pages = extract_pdf_pages(path.read_bytes())
        if any(page["text"].strip() for page in pages):
            text_pdfs.append(path)
    if len(text_pdfs) != len(pdf_paths):
        missing = sorted(set(pdf_paths) - set(text_pdfs))
        raise RuntimeError(
            "These PDFs contain no extractable text: "
            + ", ".join(path.name for path in missing)
        )
    return text_pdfs


def _load_manifest(path: Path) -> List[Dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Chunk manifest must contain a non-empty list: {path}")
    for row in rows:
        if not all(key in row for key in ("doc", "chunk_index", "text")):
            raise ValueError(f"Invalid chunk manifest row in {path}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"Empty chunk in manifest: {path}")
    return rows


def _make_schedule(chunks: List[Dict], rng: random.Random) -> List[tuple[str, List[Dict]]]:
    by_doc: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["doc"]].append(chunk)
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda chunk: chunk["chunk_index"])
    docs = sorted(by_doc)
    if len(docs) < 3:
        raise ValueError("The manifest must contain chunks from at least three documents.")

    schedule: List[tuple[str, List[Dict]]] = []

    # Five single-chunk questions per document, balancing the 20 rows.
    for doc in docs:
        available = by_doc[doc]
        selected = rng.sample(available, min(QUESTIONS_PER_DOC, len(available)))
        while len(selected) < QUESTIONS_PER_DOC:
            selected.append(rng.choice(available))
        rng.shuffle(selected)
        schedule.extend(("single", [chunk]) for chunk in selected)

    # Distribute ten same-document pairs as evenly as possible.
    pair_docs = docs[:]
    rng.shuffle(pair_docs)
    multi_counts = {doc: 10 // len(docs) for doc in docs}
    for doc in pair_docs[: 10 % len(docs)]:
        multi_counts[doc] += 1
    for doc in docs:
        pairs = list(combinations(by_doc[doc], 2))
        if not pairs and multi_counts[doc]:
            raise ValueError(f"Not enough chunks for multi questions in {doc}.")
        rng.shuffle(pairs)
        for index in range(multi_counts[doc]):
            schedule.append(("multi", list(pairs[index % len(pairs)])))

    # Choose five unique windows of three adjacent chunks from one document.
    windows = []
    for doc in docs:
        indexed = {chunk["chunk_index"]: chunk for chunk in by_doc[doc]}
        for start in sorted(indexed):
            if all(start + offset in indexed for offset in range(3)):
                windows.append(
                    [indexed[start], indexed[start + 1], indexed[start + 2]]
                )
    if len(windows) < 5:
        raise ValueError(
            f"Need five adjacent three-chunk windows; found {len(windows)}."
        )
    schedule.extend(("summary", window) for window in rng.sample(windows, 5))

    # Five plausible-but-unanswerable questions, spread across source documents.
    unanswerable_docs = docs[:]
    rng.shuffle(unanswerable_docs)
    for index in range(5):
        doc = unanswerable_docs[index % len(unanswerable_docs)]
        schedule.append(("unanswerable", [rng.choice(by_doc[doc])]))

    return schedule


def generate_testset(manifest_path: Path, output_path: Path) -> List[Dict]:
    text_pdfs = _check_docs()
    chunks = _load_manifest(manifest_path)
    manifest_docs = {row["doc"] for row in chunks}
    pdf_names = {path.name for path in text_pdfs}
    if not manifest_docs.issubset(pdf_names):
        raise ValueError(
            "The chunk manifest references PDFs not present in evaluation/docs: "
            + ", ".join(sorted(manifest_docs - pdf_names))
        )

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env or evaluation/.env.")
    client = Groq(api_key=api_key)

    rng = random.Random(eval_config.random_seed)
    schedule = _make_schedule(chunks, rng)
    seen_questions: set[str] = set()
    question_history: List[str] = []
    rows = []
    type_counts: Dict[str, int] = defaultdict(int)

    for question_type, selected_chunks in schedule:
        type_counts[question_type] += 1
        row_id = f"{question_type}-{type_counts[question_type]:03d}"
        rows.append(
            _generate_row(
                client,
                question_type,
                selected_chunks,
                seen_questions,
                question_history,
                row_id,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = generate_testset(args.manifest, args.output)
    print(f"Generated {len(rows)} rows at {args.output}")
    for row in rows[:5]:
        print(json.dumps(row, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
