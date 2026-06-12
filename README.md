![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Make Retrieval Better — and Prove It

## Overview

Your RAG pipeline works. Today you make it **better** and, just as importantly, you **measure** whether your change actually helped — with numbers, not vibes. You'll improve retrieval on the same knowledge base, build a tiny evaluation set, and use an LLM-as-judge to score before and after. The headline skill is the one that separates real engineers from demo-builders: *don't claim an improvement you can't measure.*

This is lab 3 of the connected RAG track — same `knowledge_base.json`, picking up from yesterday's pipeline.

## Learning Goals

- Strengthen retrieval with hybrid search or reranking
- Build a small evaluation set and score answers along a clear dimension
- Compare two setups fairly and conclude from evidence

## Setup

Fork, clone, branch. Reuse your pipeline from lab 2 as the starting point.

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-free-gemini-key"
```

`rank-bm25` gives you keyword scoring for hybrid search; `sentence-transformers` provides a cross-encoder if you choose reranking.

> **No starter code — you build it from scratch.** There's no template in this repo; create your own working file(s) and write the code yourself. This close to the end of the bootcamp, scaffolding your own project is part of the exercise.

## Your Task

**Improve retrieval, then prove the improvement with a small evaluation.**

1. **Pick one upgrade** to your retrieval from lab 2:
   - **Hybrid search** — combine dense vector results with **BM25** keyword scores and merge the rankings, or
   - **Reranking** — retrieve a wider set (top 8), then use a cross-encoder to re-score and keep the best 3.
2. **Build a tiny eval set** of about **5 questions** over the knowledge base, each paired with the `id` of the passage that should be retrieved (your "expected" source). Include at least one question with an exact term (like the error code `0x80070005`) that plain dense retrieval tends to fumble.
3. **Measure both setups** — your lab-2 baseline and your upgraded version — on the eval set:
   - **Retrieval hit rate:** for each question, was the expected passage in the retrieved set? (a simple, exact check, no LLM needed)
   - **Faithfulness (LLM-as-judge):** for each generated answer, ask a model to judge whether the answer is fully supported by the retrieved context (yes/no).
4. Put the results in a small **comparison table** (baseline vs upgraded, both metrics) and write 2–3 sentences: did your upgrade help, hurt, or do nothing — and does the evidence support what you expected?

A negative or flat result, honestly reported, is a perfectly good submission. The skill is the measurement.

### Optional stretch

Add a **query-rewriting** step (use the model to expand a short question before retrieving) and add it as a third column in your table.

## Submission

Commit your code and an `eval_results.md` containing your comparison table and conclusion. Open a PR and paste the link.

## Quality Bar

- A real retrieval upgrade (hybrid or reranking) is implemented, not just described
- The eval set has expected passages and at least one exact-term question
- Both setups are scored on hit rate **and** faithfulness, side by side
- The conclusion follows from the numbers, even if the upgrade didn't help
