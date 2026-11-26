# Evaluation: smoke

- Date: 2026-11-26T06:41:39.621347+00:00
- Namespace: `eval-smoke`
- Test rows: 5; RAGAS-scored: 5; excluded: 0; errors: 0
- Unanswerable rows: 0; reviewed: true

## RAGAS scores

| Metric | Mean [95% bootstrap CI] | NaNs | Affected IDs |
|---|---:|---:|---|
| Faithfulness | 1.0 [1.0, 1.0] (n=5) | 0 | — |
| ResponseRelevancy | 0.5489761220457426 [0.332772823687732, 0.7654551901595802] (n=5) | 0 | — |
| LLMContextPrecisionWithReference | 0.8666666665800001 [0.59999999994, 0.9999999999] (n=5) | 0 | — |
| LLMContextRecall | 1.0 [1.0, 1.0] (n=5) | 0 | — |

## Retrieval

Single rows (k=4): hit@k 1.0 [1.0, 1.0] (n=5); MRR 0.8666666666666668 [0.6, 1.0] (n=5).

Multi rows: gold coverage n/a (n=0).

## Abstention

| Measure | Count | Rate |
|---|---:|---:|
| Unanswerable abstained | 0/0 | n/a |
| False abstentions on answerable | 0/5 | 0.0 |

## Latency

| Stage | p50 ms | p95 ms | n |
|---|---:|---:|---:|
| retrieval_ms | 370.11690001236275 | 483.3762600028422 | 5 |
| generation_ms | 499.9115999962669 | 19007.353260001397 | 5 |
| total_ms | 886.6319000080694 | 19483.896080002884 | 5 |

## By question type

| Type | Faithfulness | Response relevancy | Context precision | Context recall | Retrieval |
|---|---:|---:|---:|---:|---:|
| single | 1.0 [1.0, 1.0] (n=5) | 0.5489761220457426 [0.3219112156184188, 0.7760410284730666] (n=5) | 0.8666666665800001 [0.59999999994, 0.9999999999] (n=5) | 1.0 [1.0, 1.0] (n=5) | hit@k 1.0 [1.0, 1.0] (n=5); MRR 0.8666666666666668 [0.6, 1.0] (n=5) |

## Source documents

| Document | Pages | Chunks |
|---|---:|---:|
| AI_Fluency_vocabulary_cheat_sheet.pdf | 2 | 3 |
| Lec 4, week 2,Activation Functions.pdf | 6 | 6 |
| Lecture.pdf | 22 | 22 |
| Lectures.pdf | 22 | 22 |
