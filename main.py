# main.py — tiny CLI for CWM
import argparse
import os
from store.memory import MemoryStore
from encoders.factory import make_encoder

print("MAIN START")


def get_mem(path="runs/memory.json",
           N=128, eta=0.1, decay=0.25,
           encoder=None, *,
           encoder_name=None, model_name=None, device=None):
    """
    Load existing memory and ensure the attached encoder matches the snapshot's N.
    If file doesn't exist, create a fresh store with provided hyperparams.
    """
    try:
        m = MemoryStore.load(path)  # legacy snapshots won't have encoder stored

        # Build/adjust encoder to match snapshot N
        if encoder is not None:
            try:
                enc_N = getattr(encoder, "N", None)
                if enc_N is None or enc_N != m.N:
                    encoder = make_encoder(encoder_name or "char",
                                           N=m.N, model_name=model_name, device=device)
            except Exception:
                encoder = make_encoder(encoder_name or "char",
                                       N=m.N, model_name=model_name, device=device)
        else:
            encoder = make_encoder(encoder_name or "char",
                                   N=m.N, model_name=model_name, device=device)

        # Attach encoder
        try:
            m.set_encoder(encoder)
        except Exception:
            m.encoder = encoder
        return m

    except FileNotFoundError:
        # Fresh store
        try:
            return MemoryStore(N=N, eta=eta, decay=decay, encoder=encoder)
        except TypeError:
            # Older MemoryStore without encoder arg
            return MemoryStore(N=N, eta=eta, decay=decay)


def cmd_add(a):
    m = get_mem(a.path, N=a.N, eta=a.eta, decay=a.decay, encoder=a._encoder,
                encoder_name=a.encoder, model_name=a.model)
    m.add_document(a.id, a.text)
    m.save(a.path)
    print(f"added {a.id}")


def cmd_search(a):
    m = get_mem(a.path, N=a.N, eta=a.eta, decay=a.decay, encoder=a._encoder,
                encoder_name=a.encoder, model_name=a.model)
    rows, _ = m.search(a.query, topk=a.topk)
    for doc_id, text, score, strength in rows:
        print(f"{doc_id}\tscore={score:.2f}\tstr={strength:.2f}\t{text}")


def cmd_update(a):
    m = get_mem(a.path, N=a.N, eta=a.eta, decay=a.decay, encoder=a._encoder,
                encoder_name=a.encoder, model_name=a.model)
    rows, q = m.search(a.query, topk=1)
    if not rows:
        print("no docs")
        return
    top_id = rows[0][0]
    m.update_trace(top_id, q)
    m.save(a.path)
    print(f"reinforced {top_id}")


def cmd_decay(a):
    m = get_mem(a.path, N=a.N, eta=a.eta, decay=a.decay, encoder=a._encoder,
                encoder_name=a.encoder, model_name=a.model)
    for _ in range(a.steps):
        m.decay_traces()
    m.save(a.path)
    print(f"decayed {a.steps} step(s)")


def cmd_list(a):
    m = get_mem(a.path, N=a.N, eta=a.eta, decay=a.decay, encoder=a._encoder,
                encoder_name=a.encoder, model_name=a.model)
    for doc_id, (text, wave, last_used, strength) in m.store.items():
        print(f"{doc_id}\tstr={strength:.3f}\tlast_used={last_used}\t{text[:60]}")


def cmd_info(a):
    m = get_mem(a.path, N=a.N, eta=a.eta, decay=a.decay, encoder=a._encoder,
                encoder_name=a.encoder, model_name=a.model)
    print(f"path: {a.path}")
    print(f"N: {m.N}")
    print(f"eta: {m.eta}")
    print(f"decay: {m.decay}")
    print(f"step: {m.step}")
    print(f"docs: {len(m.store)}")


def cmd_bulk(a):
    # Adjust this import if your loader lives elsewhere (utils.loader).
    from encoders.loader import load_corpus

    # Validate TSV path
    if not os.path.exists(a.tsv):
        print(f"[bulk] TSV not found: {a.tsv}")
        return

    # If --truncate, start a fresh store; else load/attach encoder (auto-matching N)
    if a.truncate:
        m = MemoryStore(N=a.N, eta=a.eta, decay=a.decay, encoder=a._encoder)
        print(f"[bulk] starting fresh store (truncate enabled)")
    else:
        m = get_mem(a.path, N=a.N, eta=a.eta, decay=a.decay, encoder=a._encoder,
                    encoder_name=a.encoder, model_name=a.model)

    docs = load_corpus(a.tsv)
    added = 0
    for doc_id, text in docs:
        m.add_document(doc_id, text)
        added += 1
        if a.limit and added >= a.limit:
            break
        if a.every and added % a.every == 0:
            print(f"...added {added}")

    m.save(a.path)
    print(f"bulk added {added} docs from {a.tsv} → {a.path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="runs/memory.json")
    p.add_argument("--encoder", default="char", choices=["char", "embed"])
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--decay", type=float, default=0.25)

    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    a.add_argument("--id", required=True)
    a.add_argument("--text", required=True)
    a.set_defaults(func=cmd_add)

    s = sub.add_parser("search")
    s.add_argument("--query", required=True)
    s.add_argument("--topk", type=int, default=3)
    s.set_defaults(func=cmd_search)

    u = sub.add_parser("update")
    u.add_argument("--query", required=True)
    u.set_defaults(func=cmd_update)

    d = sub.add_parser("decay")
    d.add_argument("--steps", type=int, default=1)
    d.set_defaults(func=cmd_decay)

    l = sub.add_parser("list")
    l.set_defaults(func=cmd_list)

    b = sub.add_parser("bulk")
    b.add_argument("--tsv", required=True, help="Path to a TSV: doc_id<TAB>text")
    b.add_argument("--limit", type=int, default=None, help="Optionally cap docs ingested")
    b.add_argument("--truncate", action="store_true", help="Start fresh (ignore existing snapshot)")
    b.add_argument("--every", type=int, default=500, help="Print progress every N docs")
    b.set_defaults(func=cmd_bulk)

    i = sub.add_parser("info")
    i.set_defaults(func=cmd_info)

    args = p.parse_args()

    # Build encoder ONCE for fresh stores; for existing snapshots,
    # get_mem() will rebuild to match m.N if needed.
    args._encoder = make_encoder(args.encoder, N=args.N, model_name=args.model)

    # dispatch
    args.func(args)
