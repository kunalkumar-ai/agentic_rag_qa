import pickle
import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chunker import build_chunks
from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PATH,
    BM25_INDEX_PATH, PARENTS_PATH, DOCUMENTS,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest() -> None:
    """Read .txt files for all documents, chunk → embed → ChromaDB + BM25."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma_client.delete_collection("filings")
    except Exception:
        pass
    collection = chroma_client.create_collection("filings")

    all_parents: dict[str, str] = {}
    all_bm25_children = []

    for doc in DOCUMENTS:
        company, year, txt_path = doc["company"], doc["year"], doc["txt_path"]
        print(f"\nIngesting {company} {year} from {txt_path}...")

        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        all_chunks = build_chunks(text, company=company, year=year)
        children = [c for c in all_chunks if c.chunk_type == "child"]
        parents = {c.chunk_id: c.text for c in all_chunks if c.chunk_type == "parent"}
        print(f"  Built {len(parents)} parent sections, {len(children)} child paragraphs")

        all_parents.update(parents)

        batch_size = 100
        for i in range(0, len(children), batch_size):
            batch = children[i: i + batch_size]
            embeddings = _embed_batch([c.text for c in batch])
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "parent_id": c.parent_id,
                        "section_name": c.section_name,
                        "company": c.company,
                        "year": c.year,
                    }
                    for c in batch
                ],
            )
            print(f"  Embedded {min(i + batch_size, len(children))}/{len(children)} child chunks")

        all_bm25_children.extend(children)

    with open(PARENTS_PATH, "wb") as f:
        pickle.dump(all_parents, f)
    print(f"\nParent store saved: {len(all_parents)} total parent sections")

    tokenized = [c.text.lower().split() for c in all_bm25_children]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c.chunk_id for c in all_bm25_children],
            "chunk_texts": {c.chunk_id: c.text for c in all_bm25_children},
            "chunk_metas": {
                c.chunk_id: {
                    "parent_id": c.parent_id,
                    "section_name": c.section_name,
                    "company": c.company,
                    "year": c.year,
                }
                for c in all_bm25_children
            },
        }, f)
    print(f"BM25 index saved: {len(all_bm25_children)} total child chunks")
    print("\nIngestion complete.")


if __name__ == "__main__":
    ingest()
