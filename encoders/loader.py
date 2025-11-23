# loader.py
# Step 3: minimal corpus loader

from pathlib import Path

def load_corpus(collection_path: str):
    """
    Loads a TSV file where each line is: doc_id<TAB>text
    Returns: list of (doc_id, text)
    """
    docs = []
    with open(collection_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                doc_id, text = parts
                docs.append((doc_id, text))
    return docs

if __name__ == "__main__":
    # Make a tiny dummy file first if none exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    dummy_path = data_dir / "collection.tsv"

    if not dummy_path.exists():
        with open(dummy_path, "w", encoding="utf-8") as f:
            f.write("D1\tHello world document\n")
            f.write("D2\tComplex Wave Memory test\n")

    docs = load_corpus(str(dummy_path))
    print("Loaded documents:", docs)
