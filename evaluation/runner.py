# evaluation/runner.py
# Evaluation harness: loads a corpus, builds memory, runs search, computes metrics.

from __future__ import annotations
from typing import Dict, List, Tuple
import argparse
import time
import numpy as np
from tqdm import tqdm


# local modules
from store.memory import MemoryStore
from encoders.factory import make_encoder
import evaluation.metrics as metrics  # mrr_at_10, ndcg_at_10, recall_at_k

# optional: FAISS + SentenceTransformers for shortlist
import faiss
from sentence_transformers import SentenceTransformer


# ---------- loaders ----------

def load_collection(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                doc_id, text = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                doc_id, text = parts
            out.append((doc_id, text))
    return out


def load_queries(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                qid, text = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                qid, text = parts
            out.append((qid, text))
    return out


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                qid, _zero, doc_id, rel = parts[:4]
            elif "\t" in line:
                parts = line.split("\t")
                if len(parts) == 3:
                    qid, doc_id, rel = parts
                else:
                    continue
            else:
                continue
            rel = int(rel)
            qrels.setdefault(qid, {})
            qrels[qid][doc_id] = max(rel, qrels[qid].get(doc_id, 0))
    return qrels


# ---------- eval core ----------

def build_memory(docs: List[Tuple[str, str]],
                 encoder_name: str,
                 N: int,
                 eta: float,
                 decay: float,
                 model_name: str | None,
                 device: str | None) -> MemoryStore:
    enc = make_encoder(name=encoder_name, N=N, model_name=model_name, device=device)
    mem = MemoryStore(N=N, eta=eta, decay=decay, encoder=enc)
    for doc_id, text in docs:
        mem.add_document(doc_id, text, strength=1.0)
    return mem


from tqdm import tqdm

def run_search(mem: MemoryStore,
               queries: List[Tuple[str, str]],
               topk: int,
               K: int,
               lam: float,
               shortlist: int | None = None,
               faiss_index=None,
               st_model=None,
               doc_ids=None) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, float]]:

    ranked: Dict[str, List[Tuple[str, float]]] = {}
    latencies = []

    # wrap queries with tqdm
    for qid, qtext in tqdm(queries, desc="Running queries", unit="q"):
        start = time.time()

        # Stage 1: shortlist with FAISS (if enabled)
        if shortlist and faiss_index is not None:
            q_emb = st_model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)
            D, I = faiss_index.search(q_emb, shortlist)
            candidate_ids = set(doc_ids[i] for i in I[0])
        else:
            candidate_ids = None  # full scan

        # Stage 2: CWM resonance search
        rows, _q = mem.search(qtext, topk=topk, K=K, lam=lam, restrict_ids=candidate_ids)

        end = time.time()

        hits = [(doc_id, score) for (doc_id, _text, score, _strength) in rows]
        hits = sorted(hits, key=lambda x: (-x[1], x[0]))
        ranked[qid] = hits

        latencies.append((end - start) * 1000.0)  # ms

    latency_stats = {
        "mean": float(np.mean(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    }
    return ranked, latency_stats


def main():
    ap = argparse.ArgumentParser("CWM eval runner")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--queries",    required=True)
    ap.add_argument("--qrels",      required=True)
    ap.add_argument("--encoder",    default="char", choices=["char", "embed"])
    ap.add_argument("--model",      default="all-MiniLM-L6-v2")
    ap.add_argument("--device",     default=None)
    ap.add_argument("--N",          type=int, default=1024)
    ap.add_argument("--eta",        type=float, default=0.10)
    ap.add_argument("--decay",      type=float, default=0.0)
    ap.add_argument("--topk",       type=int, default=10)
    ap.add_argument("--K",          type=int, default=128)
    ap.add_argument("--lam",        type=float, default=1.0)
    ap.add_argument("--shortlist",  type=int, default=None,
                    help="If set, use FAISS to shortlist this many candidates before CWM re-ranking")
    args = ap.parse_args()

    # --- load data ---
    docs    = load_collection(args.collection)
    queries = load_queries(args.queries)
    qrels   = load_qrels(args.qrels)

    # --- build memory ---
    mem = build_memory(docs, args.encoder, args.N, args.eta, args.decay, args.model, args.device)

    # --- optional FAISS index ---
    faiss_index = None
    st_model = None
    doc_ids = [doc_id for (doc_id, _text) in docs]

    if args.shortlist and args.shortlist > 0:
        import os
        from tqdm import tqdm

        index_path = "data/cwm_index.faiss"

        if os.path.exists(index_path):
            print(f"Loading existing FAISS index from {index_path}...")
            faiss_index = faiss.read_index(index_path)
            st_model = SentenceTransformer(args.model, device=args.device)
        else:
            print(f"Building FAISS index over {len(docs)} docs...")
            st_model = SentenceTransformer(args.model, device=args.device)

            batch_size = 512
            all_embs = []
            for i in tqdm(range(0, len(docs), batch_size), desc="Encoding docs"):
                batch_texts = [text for (_id, text) in docs[i:i+batch_size]]
                batch_embs = st_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                all_embs.append(batch_embs)

            doc_embs = np.vstack(all_embs)

            dim = doc_embs.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(doc_embs)

            print(f"Saving FAISS index to {index_path}...")
            faiss.write_index(faiss_index, index_path)

    # --- run search (ranked results + latency stats) ---
    ranked, latency_stats = run_search(mem, queries,
                                       topk=args.topk, K=args.K, lam=args.lam,
                                       shortlist=args.shortlist,
                                       faiss_index=faiss_index,
                                       st_model=st_model,
                                       doc_ids=doc_ids)

    # --- compute metrics ---
    mrr  = metrics.mrr_at_10(ranked, qrels)
    ndcg = metrics.ndcg_at_10(ranked, qrels)
    r10  = metrics.recall_at_k(ranked, qrels, k=10)
    r100 = metrics.recall_at_k(ranked, qrels, k=100)

    # --- report results ---
    print(f"MRR@10    : {mrr:.4f}")
    print(f"nDCG@10   : {ndcg:.4f}")
    print(f"Recall@10 : {r10:.4f}")
    print(f"Recall@100: {r100:.4f}")

    print(f"Latency mean={latency_stats['mean']:.2f} ms, "
          f"p50={latency_stats['p50']:.2f} ms, "
          f"p95={latency_stats['p95']:.2f} ms")

    # --- memory footprint ---
    if len(mem.store) > 0:
        first_doc = next(iter(mem.store.values()))
        text, wave, last_used, strength = first_doc
        bytes_per_entry = wave.nbytes
        total_bytes = bytes_per_entry * len(mem.store)
        print(f"Memory per entry: {bytes_per_entry/1024:.2f} KB")
        print(f"Total memory: {total_bytes/(1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
