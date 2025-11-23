# make_trec_data_judged.py
# Export judged-relevant docs + negatives to fixed ~100k size

import os, uuid, gzip, io, requests, random
from pathlib import Path
from collections import defaultdict
import ir_datasets
from tqdm import tqdm

os.environ["PYTHONUTF8"] = "1"
tmpdir = os.path.join(r"E:\\cwm\\.tmp", "run_" + str(uuid.uuid4()))
os.makedirs(tmpdir, exist_ok=True)
os.environ["TEMP"] = tmpdir
os.environ["TMP"]  = tmpdir

# Official queries
QUERIES_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz"

# Change data set size here:
def export_trec_dl_judged(target_size: int = 1000):
    out = Path("data"); out.mkdir(exist_ok=True)

    # --- Step 1: load judged qrels ---
    ds_judged = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
    qrels = defaultdict(list)
    for qr in ds_judged.qrels_iter():
        if qr.relevance > 0:
            qrels[qr.query_id].append(qr.doc_id)
    kept_qids = set(qrels.keys())

    # --- Step 2: download the official queries ---
    resp = requests.get(QUERIES_URL, timeout=60)
    resp.raise_for_status()
    gz = gzip.GzipFile(fileobj=io.BytesIO(resp.content))
    with io.TextIOWrapper(gz, encoding="utf-8", newline="") as gzf, \
         open(out / "queries.tsv", "w", encoding="utf-8", newline="") as qout:
        for line in gzf:
            qid, text = line.strip().split("\t", 1)
            if qid in kept_qids:
                qout.write(f"{qid}\t{text}\n")

    # --- Step 3: collect relevant docs ---
    relevant_ids = {doc_id for docs in qrels.values() for doc_id in docs}
    docs = {}

    print(f"Collecting relevant docs ({len(relevant_ids):,})...")
    ds_all = ir_datasets.load("msmarco-passage")
    for d in tqdm(ds_all.docs_iter(), total=ds_all.docs_count(), desc="Collecting relevant"):
        if d.doc_id in relevant_ids:
            docs[d.doc_id] = d.text
        if len(docs) == len(relevant_ids):
            break

    # --- Step 4: sample negatives until target size ---
    needed_neg = target_size - len(docs)
    print(f"Sampling {needed_neg:,} negatives...")
    negatives = []
    for d in tqdm(ds_all.docs_iter(), total=ds_all.docs_count(), desc="Sampling negatives"):
        if d.doc_id not in docs and d.doc_id not in relevant_ids:
            negatives.append((d.doc_id, d.text))
            if len(negatives) >= needed_neg:
                break

    for doc_id, text in negatives:
        docs[doc_id] = text

    # --- Step 5: write collection ---
    with open(out / "collection.tsv", "w", encoding="utf-8", newline="") as f:
        for doc_id, text in docs.items():
            f.write(f"{doc_id}\t{text}\n")

    # --- Step 6: write qrels ---
    with open(out / "qrels.txt", "w", encoding="utf-8", newline="") as f:
        for qid, rel_docs in qrels.items():
            for doc_id in rel_docs:
                f.write(f"{qid} 0 {doc_id} 1\n")

    print(f"Export complete → data/ (docs={len(docs):,}, queries={len(kept_qids)})")

if __name__ == "__main__":
    print("Starting export (judged slice, capped.)...")
    export_trec_dl_judged()
    print("Done.")
