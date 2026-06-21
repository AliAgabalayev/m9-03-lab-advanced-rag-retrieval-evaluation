# Evaluation Results — Hybrid Retrieval vs Dense Baseline

**Upgrade chosen:** hybrid search — dense (Chroma, MiniLM) fused with **BM25**
keyword scores via Reciprocal Rank Fusion (RRF). Reproduce with
`python advanced_rag_eval.py`.

## Eval set (question → expected passage)

| # | Question | Expected | Note |
|---|----------|----------|------|
| 1 | How long do I have to get a full refund? | kb-04 | |
| 2 | How do I reset my password? | kb-07 | |
| 3 | What does error 0x80070005 mean? | kb-08 | **exact-term** |
| 4 | When can employees park in lot B? | kb-01 | |
| 5 | How do I cancel my subscription? | kb-05 | |

## Retrieval hit rate (expected id in top-3)

| Question | exp | baseline top-3 | hybrid top-3 | base hit | hybrid hit |
|----------|-----|----------------|--------------|:---:|:---:|
| refund | kb-04 | kb-04, kb-05, kb-06 | kb-04, kb-06, kb-05 | ✅ | ✅ |
| password | kb-07 | kb-07, kb-02, kb-05 | kb-07, kb-02, kb-05 | ✅ | ✅ |
| 0x80070005 | kb-08 | kb-08, kb-02, kb-07 | kb-08, kb-02, kb-01 | ✅ | ✅ |
| park lot B | kb-01 | kb-01, kb-10, kb-03 | kb-01, kb-03, kb-06 | ✅ | ✅ |
| cancel sub | kb-05 | kb-05, kb-07, kb-06 | kb-05, kb-02, kb-07 | ✅ | ✅ |

## Comparison table

| Metric | Baseline (dense) | Upgrade (hybrid) |
|--------|:---:|:---:|
| **Retrieval hit rate** | 100% (5/5) | 100% (5/5) |
| **Faithfulness (LLM judge)** | see note | see note |

> **Faithfulness** runs an LLM-as-judge (generate an answer per setup, then ask
> Gemini YES/NO "is the answer fully supported by the retrieved context"). It is
> skipped in the captured run because no `GOOGLE_API_KEY` was set. Since every
> answer is grounded strictly in the retrieved passages by the prompt, both
> setups are expected to score ~100% faithful — faithfulness is bounded by the
> prompt's grounding, not by which retriever fed it. Export a key and re-run to
> fill in the exact numbers.

## Conclusion (2–3 sentences)

On this 10-passage corpus the upgrade was a **flat result**: hybrid retrieval
neither helped nor hurt — both setups put the expected passage in the top-3 for
all 5 questions, including the exact-term `0x80070005` case where I *expected*
dense to fumble. It didn't fumble, because kb-08 literally says "access denied"
and the surrounding semantics already pull it to rank 1, so BM25 had nothing to
rescue. The honest takeaway is that BM25's exact-match advantage only shows up
when the corpus is larger or the query term is rare/out-of-vocabulary for the
embedder; here the measurement proves the change was safe but unnecessary — which
is exactly the point: don't claim an improvement the numbers don't support.
