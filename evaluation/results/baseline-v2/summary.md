# Evaluation: baseline-v2

- Date: 2026-11-26T07:53:32.974766+00:00
- Namespace: `eval-baseline-v2`
- Test rows: 40; RAGAS-scored: 20; excluded: 5; errors: 0
- Unanswerable rows: 5; reviewed: true

## Warnings

- Judge model appears in the configured generator fallback chain; evaluation may be self-biased.

## RAGAS scores

| Metric | Mean [95% bootstrap CI] | NaNs | Affected IDs |
|---|---:|---:|---|
| Faithfulness | 0.8444444444444444 [0.6444444444444444, 1.0] (n=15) | 20 | single-016, single-017, single-018, single-019, single-020, multi-001, multi-002, multi-003, multi-004, multi-005, multi-006, multi-007, multi-008, multi-009, multi-010, summary-001, summary-002, summary-003, summary-004, summary-005 |
| ResponseRelevancy | 0.3662532455190764 [0.22793700161602246, 0.5055313355500539] (n=20) | 15 | single-017, single-018, single-019, single-020, multi-001, multi-003, multi-004, multi-005, multi-007, multi-008, multi-009, multi-010, summary-002, summary-003, summary-004 |
| LLMContextPrecisionWithReference | 0.8888888888161112 [0.7111111110571529, 0.9999999999238888] (n=15) | 20 | single-016, single-017, single-018, single-019, single-020, multi-001, multi-002, multi-003, multi-004, multi-005, multi-006, multi-007, multi-008, multi-009, multi-010, summary-001, summary-002, summary-003, summary-004, summary-005 |
| LLMContextRecall | 0.9230769230769231 [0.7692307692307693, 1.0] (n=13) | 22 | single-004, single-014, single-016, single-017, single-018, single-019, single-020, multi-001, multi-002, multi-003, multi-004, multi-005, multi-006, multi-007, multi-008, multi-009, multi-010, summary-001, summary-002, summary-003, summary-004, summary-005 |

## Retrieval

Single rows (k=4): hit@k 0.95 [0.85, 1.0] (n=20); MRR 0.9166666666666666 [0.8, 1.0] (n=20).

Multi rows: gold coverage 0.85 [0.7, 1.0] (n=10).

## Abstention

| Measure | Count | Rate |
|---|---:|---:|
| Unanswerable abstained | 5/5 | 1.0 |
| False abstentions on answerable | 10/35 | 0.2857142857142857 |

## Latency

| Stage | p50 ms | p95 ms | n |
|---|---:|---:|---:|
| retrieval_ms | 348.50350000488106 | 500.07143498805806 | 40 |
| generation_ms | 817.5304499891354 | 4763.874195000968 | 40 |
| total_ms | 1186.1798000027193 | 5061.948025012676 | 40 |

## By question type

| Type | Faithfulness | Response relevancy | Context precision | Context recall | Retrieval |
|---|---:|---:|---:|---:|---:|
| multi | n/a (n=0) | 0.0 [0.0, 0.0] (n=2) | n/a (n=0) | n/a (n=0) | 0.85 [0.7, 0.95] (n=10) |
| single | 0.8444444444444444 [0.6444444444444444, 1.0] (n=15) | 0.4578165568988455 [0.3064060247226963, 0.6089270039528784] (n=16) | 0.8888888888161112 [0.7333333332716528, 0.9999999999222222] (n=15) | 0.9230769230769231 [0.7692307692307693, 1.0] (n=13) | hit@k 0.95 [0.85, 1.0] (n=20); MRR 0.9166666666666666 [0.7995833333333334, 1.0] (n=20) |
| summary | n/a (n=0) | 0.0 [0.0, 0.0] (n=2) | n/a (n=0) | n/a (n=0) | — |
| unanswerable | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | — |

## Source documents

| Document | Pages | Chunks |
|---|---:|---:|
| AI_Fluency_vocabulary_cheat_sheet.pdf | 2 | 3 |
| Lec 4, week 2,Activation Functions.pdf | 6 | 6 |
| Lecture.pdf | 22 | 22 |
| Lectures.pdf | 22 | 22 |
