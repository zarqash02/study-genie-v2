# Study Genie RAGAS Evaluation

This self-contained harness measures Study Genie's retrieval and answer quality
against PDFs using the app's own extraction, chunking, embeddings, retrieval,
and generation pipeline. RAGAS scores answer quality with a configured LLM
judge. Place source PDFs in `docs/`; evaluation artifacts are written to
`cache/` and `results/`.

## Safety rules

- Evaluation Pinecone vectors use a namespace beginning with `eval-`.
- Destructive evaluation operations must verify the namespace prefix before deletion.
- Evaluation scripts load the app's root `.env` and `evaluation/.env`.
- Never place API keys in source files or commit `evaluation/.env`.

## Setup

Use the repository's Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r evaluation\requirements-eval.txt
Copy-Item evaluation\.env.example evaluation\.env
```

Set `GROQ_API_KEY` and `EVAL_JUDGE_MODEL` in `evaluation/.env`. The judge model
must be available through the configured Groq-compatible API. The example uses
`openai/gpt-oss-120b`. Set the app's required Pinecone and Hugging Face
credentials and settings in the root `.env`; evaluation loads those existing
app settings as well. Never commit either `.env` file.

## Adding documents

Put 3-5 text-based PDFs in `evaluation/docs/`. The test-set generator checks
that these PDFs have extractable text; scanned PDFs without text extraction
will not work. The harness does not download documents. Ingest them into an
evaluation-only namespace and write the chunk manifest without running a RAG
question:

```powershell
.venv\Scripts\python.exe -c "from evaluation.rag_runner import ingest_eval_docs; print(ingest_eval_docs('eval-testset'))"
```

Ingestion reuses the app's PDF extraction, chunking, embedding, and Pinecone
services. Namespaces must start with `eval-`; do not point evaluation at a
production namespace. Existing evaluation vectors are reused unless you
explicitly request reingestion.

## Test-set generation and review

After ingestion has created `evaluation/cache/eval-testset_chunks.json`, generate
a draft:

```powershell
.venv\Scripts\python.exe -m evaluation.generate_testset
```

The generator checks for 3-5 text-extractable PDFs, then writes
`evaluation/testset_draft.json` with 20 single-chunk, 10 same-document
multi-chunk, 5 adjacent-chunk summary, and 5 unanswerable questions. The Groq
judge creates questions at temperature 0.3; its references must use only the
provided chunks. Draft rows are marked `reviewed: false`.

Review **every row**. Correct or delete bad rows, set `reviewed` to `true`, and
save the reviewed file as `evaluation/testset.json`. `run_eval.py` refuses the
draft filename or any unreviewed row by default. `--allow-unreviewed` explicitly
overrides that guard and prints `UNREVIEWED TEST SET`.

## Running evaluations

Run a small smoke test, then evaluate the reviewed test set:

```powershell
.venv\Scripts\python.exe -m evaluation.run_eval --tag smoke --limit 5 --allow-unreviewed
.venv\Scripts\python.exe -m evaluation.run_eval --tag baseline
```

The namespace defaults to `eval-<tag>`. Outputs and scores are cached by tag so
interrupted runs can resume. To compare saved runs:

```powershell
.venv\Scripts\python.exe -m evaluation.compare
```

Useful `run_eval` options:

- `--namespace NAME`: choose an evaluation namespace (must begin `eval-`).
- `--top-k N`: override the app's configured retrieval count.
- `--testset PATH`: use a different test-set JSON file.
- `--reingest`: replace/recreate the evaluation namespace's vectors and manifest.
- `--limit N`: run only the first N test-set rows.
- `--skip-generation` / `--skip-scoring`: reuse cached outputs or skip RAGAS scoring.
- `--max-workers N`: scoring worker count (default `2`).
- `--sleep-seconds N`: delay between generation calls (default `1`).
- `--allow-unreviewed`: explicitly permit draft or unreviewed rows.
- `--cleanup`: delete vectors after the run; accepted only for an `eval-` namespace.

## Metrics

- **Faithfulness:** how well the generated answer is supported by retrieved context.
- **Response relevancy:** how relevant the answer is to the question.
- **Context precision:** how much of the retrieved context is relevant to the reference.
- **Context recall:** how much of the reference is supported by retrieved context.
- **Hit@k:** whether the gold chunk appears in the top-k retrieved contexts for single-chunk questions.
- **MRR:** reciprocal rank of the gold chunk, or zero when it is not retrieved.
- **Gold coverage:** fraction of gold chunks retrieved for multi-chunk questions.
- **Abstention rate:** fraction of unanswerable questions where the app refuses; false abstention is reported separately for answerable rows.
- **Latency:** retrieval, generation, and total elapsed time, summarized by p50 and p95.

RAGAS scoring excludes unanswerable rows and answerable rows with generation
errors or empty answers. NaNs are reported rather than imputed.

## Limitations

- The test set is small, so estimates and confidence intervals can be unstable.
- LLM-as-judge scores have prompt and model noise; the judge is not a human evaluator.
- The evaluated PDFs cover a limited document domain and may not represent other subjects or formats.
- Free-tier generator and judge models can have rate limits, availability changes, and quality constraints.
