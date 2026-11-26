# Evaluation: baseline

- Date: 2026-11-26T06:40:24.138641+00:00
- Namespace: `eval-baseline`
- Test rows: 40; RAGAS-scored: 35; excluded: 5; errors: 0
- Unanswerable rows: 5; reviewed: true

## RAGAS scores

| Metric | Mean [95% bootstrap CI] | NaNs | Affected IDs |
|---|---:|---:|---|
| Faithfulness | 0.7999361282974728 [0.6887149359743898, 0.8965596276972327] (n=35) | 0 | — |
| ResponseRelevancy | 0.6596800986217825 [0.5656282133585868, 0.7467390328594705] (n=35) | 0 | — |
| LLMContextPrecisionWithReference | 0.8547619047012303 [0.7571428570910753, 0.933333333269339] (n=35) | 0 | — |
| LLMContextRecall | 0.8785714285714286 [0.7642857142857142, 0.9714285714285714] (n=35) | 0 | — |

## Retrieval

Single rows (k=4): hit@k 0.95 [0.85, 1.0] (n=20); MRR 0.9166666666666666 [0.8, 1.0] (n=20).

Multi rows: gold coverage 0.85 [0.7, 1.0] (n=10).

## Abstention

| Measure | Count | Rate |
|---|---:|---:|
| Unanswerable abstained | 4/5 | 0.8 |
| False abstentions on answerable | 1/35 | 0.02857142857142857 |

## Latency

| Stage | p50 ms | p95 ms | n |
|---|---:|---:|---:|
| retrieval_ms | 342.5167499954114 | 1701.9897050042327 | 40 |
| generation_ms | 4259.656200003519 | 14528.303635001792 | 40 |
| total_ms | 4517.894349999551 | 15016.919425001832 | 40 |

## By question type

| Type | Faithfulness | Response relevancy | Context precision | Context recall | Retrieval |
|---|---:|---:|---:|---:|---:|
| multi | 0.6251610644257702 [0.45539093137254905, 0.7876206232492996] (n=10) | 0.8511056596413313 [0.7970970313471182, 0.9041511673061561] (n=10) | 0.7916666666250001 [0.6249999999649374, 0.9499999999392499] (n=10) | 0.8 [0.5, 1.0] (n=10) | 0.85 [0.7, 0.95] (n=10) |
| single | 0.8875 [0.7375, 1.0] (n=20) | 0.5318315322684054 [0.4062811184984642, 0.6544218108351729] (n=20) | 0.9083333332591665 [0.783333333275375, 0.999999999915833] (n=20) | 0.95 [0.85, 1.0] (n=20) | hit@k 0.95 [0.85, 1.0] (n=20); MRR 0.9166666666666666 [0.8, 1.0] (n=20) |
| summary | 0.7992307692307692 [0.59, 0.9692307692307691] (n=5) | 0.788223241996193 [0.7630470503656601, 0.8124623065590004] (n=5) | 0.7666666666219444 [0.4833333333047222, 0.9499999999416666] (n=5) | 0.75 [0.35, 1.0] (n=5) | — |
| unanswerable | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | — |

## Source documents

| Document | Pages | Chunks |
|---|---:|---:|
| AI_Fluency_vocabulary_cheat_sheet.pdf | 2 | 3 |
| Lec 4, week 2,Activation Functions.pdf | 6 | 6 |
| Lecture.pdf | 22 | 22 |
| Lectures.pdf | 22 | 22 |
