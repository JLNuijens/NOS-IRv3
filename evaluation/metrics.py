from typing import Dict, List, Tuple
import math

__all__ = ["mrr_at_10", "ndcg_at_10", "recall_at_k"]

def mrr_at_10(ranked: Dict[str, List[Tuple[str, float]]],
              qrels: Dict[str, Dict[str, int]]) -> float:
    total, count = 0.0, 0
    for qid, results in ranked.items():
        rels = qrels.get(qid, {})
        rr = 0.0
        for i, (doc_id, _score) in enumerate(results[:10], start=1):
            if rels.get(doc_id, 0) > 0:
                rr = 1.0 / i
                break
        total += rr
        count += 1
    return total / count if count else 0.0

def ndcg_at_10(ranked: Dict[str, List[Tuple[str, float]]],
               qrels: Dict[str, Dict[str, int]]) -> float:
    def dcg(gains: List[int]) -> float:
        return sum(((2**rel - 1) / math.log2(i + 2)) for i, rel in enumerate(gains))
    total, count = 0.0, 0
    for qid, results in ranked.items():
        rels = qrels.get(qid, {})
        gains = [rels.get(doc_id, 0) for doc_id, _ in results[:10]]
        dcg_val = dcg(gains)
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg_val = dcg(ideal) if ideal else 0.0
        total += (dcg_val / idcg_val) if idcg_val > 0 else 0.0
        count += 1
    return total / count if count else 0.0

def recall_at_k(ranked: Dict[str, List[Tuple[str, float]]],
                qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    total, count = 0.0, 0
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        retrieved = {doc_id for doc_id, _ in ranked.get(qid, [])[:k]}
        total += len(relevant & retrieved) / len(relevant)
        count += 1
    return total / count if count else 0.0
