"""
Lab | Make Retrieval Better — and Prove It
==========================================

Upgrade retrieval over the same knowledge base, then MEASURE whether it helped.

  * Baseline  : dense vector retrieval (Chroma), top-3.
  * Upgrade   : HYBRID search = dense + BM25 keyword scores fused with
                Reciprocal Rank Fusion (RRF), top-3.  Chosen because exact
                terms like the error code `0x80070005` are exactly what dense
                retrieval fumbles and BM25 nails.

Metrics on a 5-question eval set:
  * Retrieval hit rate  — was the expected passage id in the top-3? (no LLM)
  * Faithfulness        — LLM-as-judge: is the answer fully supported by the
                          retrieved context? (yes/no; needs GOOGLE_API_KEY)

Run:  python advanced_rag_eval.py   (hit rate runs keyless; faithfulness needs a key)
"""

import json
import os
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

KB_PATH = Path(__file__).with_name("knowledge_base.json")
GEN_MODEL = "gemini-2.5-flash"

# Eval set: question -> expected passage id. Includes an exact-term question
# (0x80070005) that plain dense retrieval tends to fumble.
EVAL_SET = [
    {"q": "How long do I have to get a full refund?", "expected": "kb-04"},
    {"q": "How do I reset my password?", "expected": "kb-07"},
    {"q": "What does error 0x80070005 mean?", "expected": "kb-08"},  # exact term
    {"q": "When can employees park in lot B?", "expected": "kb-01"},
    {"q": "How do I cancel my subscription?", "expected": "kb-05"},
]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Retrievers:
    """Holds the KB plus a dense (Chroma) and a BM25 index over it."""

    def __init__(self):
        self.kb = json.loads(KB_PATH.read_text())
        self.ids = [e["id"] for e in self.kb]
        self.by_id = {e["id"]: e for e in self.kb}

        # Dense index (keyless local embeddings unless a key is present).
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=api_key, model_name="models/gemini-embedding-001"
            )
        else:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        client = chromadb.Client()
        try:
            client.delete_collection("kb3")
        except Exception:
            pass
        self.collection = client.create_collection(
            "kb3", embedding_function=ef, metadata={"hnsw:space": "cosine"}
        )
        self.collection.add(
            ids=self.ids,
            documents=[e["text"] for e in self.kb],
            metadatas=[{"source": e["source"]} for e in self.kb],
        )

        # BM25 index.
        self.bm25 = BM25Okapi([tokenize(e["text"]) for e in self.kb])

    # --- Baseline: dense only ------------------------------------------------
    def dense(self, question, k=3):
        res = self.collection.query(query_texts=[question], n_results=k)
        return res["ids"][0]

    # --- Upgrade: hybrid dense + BM25 via Reciprocal Rank Fusion --------------
    def hybrid(self, question, k=3, pool=8, rrf_k=60):
        # Dense ranking (wider pool).
        dense_ranked = self.collection.query(
            query_texts=[question], n_results=pool
        )["ids"][0]
        # BM25 ranking.
        scores = self.bm25.get_scores(tokenize(question))
        bm25_ranked = [
            self.ids[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])
        ][:pool]

        # Fuse: RRF score = sum over lists of 1/(rrf_k + rank).
        fused = {}
        for ranked in (dense_ranked, bm25_ranked):
            for rank, _id in enumerate(ranked):
                fused[_id] = fused.get(_id, 0.0) + 1.0 / (rrf_k + rank)
        return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])][:k]


# --------------------------------------------------------------------------- #
# Generation + LLM-as-judge faithfulness (need a key).
# --------------------------------------------------------------------------- #
def _client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    from google import genai

    return genai.Client(api_key=api_key)


def generate_answer(client, retr, question, ids):
    context = "\n".join(f"[{retr.by_id[i]['source']}] {retr.by_id[i]['text']}" for i in ids)
    prompt = (
        "Answer using ONLY the context. If absent, say \"I don't know\". "
        f"Cite the source.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    return client.models.generate_content(model=GEN_MODEL, contents=prompt).text.strip()


def judge_faithful(client, retr, question, ids, answer):
    context = "\n".join(retr.by_id[i]["text"] for i in ids)
    prompt = (
        "You are a strict judge. Is the ANSWER fully supported by the CONTEXT "
        "(no facts beyond it)? Reply with a single word: YES or NO.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}\n\nVERDICT:"
    )
    verdict = client.models.generate_content(model=GEN_MODEL, contents=prompt).text
    return "YES" if "yes" in verdict.strip().lower()[:5] else "NO"


def main():
    retr = Retrievers()
    client = _client()

    rows = []
    for item in EVAL_SET:
        q, expected = item["q"], item["expected"]
        baseline_ids = retr.dense(q, k=3)
        upgrade_ids = retr.hybrid(q, k=3)

        row = {
            "q": q,
            "expected": expected,
            "baseline_ids": baseline_ids,
            "upgrade_ids": upgrade_ids,
            "baseline_hit": expected in baseline_ids,
            "upgrade_hit": expected in upgrade_ids,
            "baseline_faithful": None,
            "upgrade_faithful": None,
        }
        if client:
            for tag, ids in (("baseline", baseline_ids), ("upgrade", upgrade_ids)):
                ans = generate_answer(client, retr, q, ids)
                row[f"{tag}_faithful"] = judge_faithful(client, retr, q, ids, ans)
        rows.append(row)

    # Print a comparison table.
    print(f"{'question':42} | exp | base hit | up hit | base ids / up ids")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['q'][:42]:42} | {r['expected']} | "
            f"{str(r['baseline_hit']):8} | {str(r['upgrade_hit']):6} | "
            f"{r['baseline_ids']} / {r['upgrade_ids']}"
        )

    b_hit = sum(r["baseline_hit"] for r in rows) / len(rows)
    u_hit = sum(r["upgrade_hit"] for r in rows) / len(rows)
    print(f"\nHit rate  — baseline: {b_hit:.0%}   hybrid: {u_hit:.0%}")
    if client:
        b_f = sum(r["baseline_faithful"] == "YES" for r in rows) / len(rows)
        u_f = sum(r["upgrade_faithful"] == "YES" for r in rows) / len(rows)
        print(f"Faithful  — baseline: {b_f:.0%}   hybrid: {u_f:.0%}")
    else:
        print("Faithfulness skipped (no GOOGLE_API_KEY).")


if __name__ == "__main__":
    main()
