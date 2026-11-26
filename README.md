# Study Genie

Study Genie is a PDF study assistant that retrieves relevant material from uploaded documents and uses an LLM to answer questions and create study aids.

**Try it live:** https://huggingface.co/spaces/zarqash/FYP

## Features

- Upload PDF files or paste study text.
- Add documents to a session library and remove documents by ID.
- Ask questions grounded in uploaded material and receive source document, page, similarity score, and text snippet.
- Generate structured summaries.
- Extract key concepts and related terms.
- Render extracted concepts as a mind map.
- Generate a learning guide for a topic.
- Generate multiple-choice quizzes and track quiz accuracy and weak topics.
- Generate flashcards and move between card faces.
- Save summaries, learning guides, concepts, chat answers, and other notes.
- Download summaries, learning guides, and saved notes.
- Track questions asked, quizzes taken, quiz answers, and missed questions.
- Delete a document from Pinecone through the API or UI.

## Architecture

PDF upload: PyMuPDF extracts page text, the text is split into word-based chunks of 500 words with 50-word overlap, `sentence-transformers/all-MiniLM-L6-v2` creates 384-dimensional embeddings, and Pinecone stores the vectors. The `/ask` flow embeds the question, retrieves `top_k = 4` chunks, adds source metadata and retrieved text to the prompt, and sends the result to the generator chain.

Generator chain: Groq model `qwen/qwen3.8-27b` first, then Hugging Face Inference Providers in this order: `openai/gpt-oss-120b`, `zai-org/GLM-5.3`, `deepseek-ai/DeepSeek-V4.1-Flash-1017`, and `Qwen/Qwen3.5-9B`.

The prompt tells the model to answer only from the provided context and, if the answer is not in the context, to say `I don't know based on the provided material.` This refusal rule is applied to question answering.

```text
upload -> extract -> chunk -> embed -> Pinecone
question -> embed -> retrieve top-k -> prompt -> LLM -> answer
```

## Tech stack

| Component | Code and requirement |
|---|---|
| API | FastAPI; `fastapi==0.115.0` |
| UI | Streamlit `1.38.0` |
| PDF extraction | PyMuPDF `1.24.10` |
| Embeddings | `sentence-transformers` and `sentence-transformers==2.7.0` |
| Vector database | Pinecone `pinecone==5.0.1` |
| Primary generation | Groq |
| Fallback generation | Hugging Face Inference Providers through `huggingface_hub>=0.34.0` |
| Evaluation | RAGAS `0.4.4`; pandas `2.3.3`; NumPy `2.2.6` |
| Runtime | Python `3.11` Docker image |
| Configuration | python-dotenv `1.0.1` |

## Project structure

```text
study-genie/
├── backend/
│   ├── __init__.py                 # Marks the backend package
│   ├── config.py                   # Loads environment configuration
│   ├── main.py                     # Defines the FastAPI application and endpoints
│   └── services/
│       ├── __init__.py             # Marks the services package
│       ├── embedding_service.py    # Loads the local embedding model
│       ├── llm_service.py          # Builds prompts and calls the generator chain
│       ├── pdf_service.py          # Extracts PDF text and creates chunks
│       └── pinecone_service.py     # Creates, updates, queries, and deletes Pinecone vectors
├── frontend/
│   └── app.py                      # Defines the Streamlit study interface
├── evaluation/
│   ├── README.md                   # Documents evaluation setup and commands
│   ├── __init__.py                 # Marks the evaluation package
│   ├── config.py                   # Loads evaluation configuration
│   ├── generate_testset.py         # Generates a reviewed test-set draft
│   ├── rag_runner.py               # Runs retrieval and generation for evaluation
│   ├── run_eval.py                 # Runs RAGAS evaluation
│   ├── stability_check.py          # Repeats selected generations
│   ├── compare.py                  # Compares saved evaluation runs
│   ├── testset.json                # Reviewed 40-question test set
│   ├── testset_draft.json          # Draft test set
│   ├── cache/                      # Generation, chunk, and score caches
│   ├── docs/                       # Evaluation PDF documents
│   └── results/                    # Evaluation summaries and per-row results
├── Dockerfile                      # Builds the Docker image and starts `start.sh`
├── start.sh                        # Runs Streamlit on port 7860
├── requirements.txt                # Application dependencies
├── .env.example                    # Names the supported environment variables
└── README.md                       # Project documentation
```

## Run locally

Copy `.env.example` to `.env` and set the required credentials:

```text
PINECONE_API_KEY
GROQ_API_KEY
HF_TOKEN
```

At least one of `GROQ_API_KEY` or `HF_TOKEN` is required. `PINECONE_API_KEY` is also required.

Start the application with:

```text
./start.sh
```

`start.sh` runs Streamlit on `0.0.0.0:7860`.

## Docker and Hugging Face Spaces

The Dockerfile is based on `python:3.11-slim`, installs `requirements.txt`, copies the application, sets port `7860`, and runs `./start.sh`.

```text
docker build -t study-genie .
docker run -p 7860:7860 --env-file .env study-genie
```

The supported environment variable names are:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_CLOUD`
- `PINECONE_REGION`
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `HF_TOKEN`
- `HF_GENERATION_MODEL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIM`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`

## API endpoints

| Method | Path | Input | Response |
|---|---|---|---|
| `GET` | `/health` | None | `{"status": "ok"}` |
| `POST` | `/upload` | Multipart PDF file | `doc_id`, `doc_name`, `chunks` |
| `POST` | `/ask` | `query`, optional `doc_ids`, optional `top_k` | `answer`, `sources` |
| `POST` | `/summarize` | Optional `doc_ids` | `summary` |
| `POST` | `/quiz` | Optional `doc_ids`, optional `n` | `quiz` |
| `POST` | `/flashcards` | Optional `doc_ids`, optional `n` | `flashcards` |
| `DELETE` | `/document/{doc_id}` | Document ID in path | `deleted` |

## Evaluation

The setup contains 40 hand-reviewed questions over 4 course documents. The documents contain 53 chunks. The test set has 20 single-chunk questions, 10 multi-chunk questions, 5 summary questions, and 5 unanswerable questions.

The baseline run used RAGAS 0.4.4, judge model `openai/gpt-oss-120b`, primary generator model `qwen/qwen3.8-27b`, embedding model `sentence-transformers/all-MiniLM-L6-v2`, chunk size 500 with 50-word overlap, and `top_k = 4`. Run date: 2026-11-26.

### Retrieval and context quality

Stable across the baseline and second run:

| Metric | Result |
|---|---:|
| Hit@4, 20 single-chunk questions | 0.95 |
| MRR, 20 single-chunk questions | 0.92 |
| Multi-chunk gold coverage, n=10 | 0.85 |
| Context precision, n=35 | 0.85 [95% CI: 0.76, 0.93] |
| Context recall, n=35 | 0.88 [95% CI: 0.76, 0.97] |
| Retrieval p50 latency | 342.52 ms |

Context precision and context recall were identical on the 15 rows scored in both runs.

### Generation metrics, run 1 only

| Metric | Result |
|---|---:|
| Faithfulness, n=35 | 0.80 [95% CI: 0.69, 0.90] |
| Response relevancy, n=35 | 0.66 [95% CI: 0.57, 0.75] |

Generation behaviour differed in a second identical run (see below), so treat these as indicative.

### Findings

- **Over-refusal:** Run 1 refused 1/35 answerable questions; run 2, under the same settings, refused 10/35. In the stability check, the 10 rows refused in run 2 refused 27 of 30 repeat attempts, while 10 control rows refused 0 of 30. All 60 attempts were answered by the primary generator. The refusals are systematic on specific rows, not random. The cause is not known. In 3 inspected cases, the retrieved context only partly supported the reference answer, and the prompt tells the model to answer only from context. This is an observation, not a proven cause.
- **ResponseRelevancy artifact:** The median length of single-answer answers was 4.50 words. The correlation between answer length and ResponseRelevancy was 0.56. Two correct short answers with low scores are `Retrieval augmented generation (RAG)` (4 words, ResponseRelevancy -0.06) and `hammer, anvil, stirrup` (3 words, ResponseRelevancy 0.24).
- **Incomplete run 2 RAGAS scoring:** The run 2 RAGAS scoring was incomplete because the judge hit its daily token limit, so run 2 RAGAS numbers are not reported.
- **Abstention on unanswerable questions:** 4 of 5 unanswerable questions were abstained in run 1.

### Reproduce

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r evaluation\requirements-eval.txt
Copy-Item evaluation\.env.example evaluation\.env
```

Set `GROQ_API_KEY` and `EVAL_JUDGE_MODEL` in `evaluation/.env`, and set the application credentials and settings in the root `.env`.

```powershell
.venv\Scripts\python.exe -c "from evaluation.rag_runner import ingest_eval_docs; print(ingest_eval_docs('eval-testset'))"
.venv\Scripts\python.exe -m evaluation.generate_testset
.venv\Scripts\python.exe -m evaluation.run_eval --tag smoke --limit 5 --allow-unreviewed
.venv\Scripts\python.exe -m evaluation.run_eval --tag baseline
.venv\Scripts\python.exe -m evaluation.compare
```

## Limitations and next steps

- The test set is small and the corpus is 53 chunks, so retrieval numbers can be easier than they would be on a large index.
- LLM-as-judge noise is present. The judge model is also in the generator fallback chain, and it answered 0 of 40 rows in run 2.
- Generation was not deterministic between the two runs; the cause is unidentified.
- Next steps: investigate and reduce over-refusal, use a larger corpus, run repeated generations per question, and add reranking.

## 📜 License

MIT
