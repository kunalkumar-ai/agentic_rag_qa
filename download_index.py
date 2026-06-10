"""Download index files from Hugging Face Hub at app startup.

On Streamlit Cloud the repo is mounted read-only at /mount/src/agentic_rag_qa/.
Index files are downloaded to /tmp/rag_index/ which is writable.
Locally the files already exist next to the code, so download is skipped.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "kunalkumar-ai/rag-index"

# writable location on Streamlit Cloud; on local machine files live next to code
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", "."))


def download_index():
    chroma_path = INDEX_DIR / "chroma_db"
    bm25_path = INDEX_DIR / "bm25_index.pkl"
    parents_path = INDEX_DIR / "parents.pkl"

    chroma_exists = chroma_path.exists() and any(chroma_path.iterdir())
    pkls_exist = bm25_path.exists() and parents_path.exists()

    print(f"Index dir: {INDEX_DIR.resolve()}")
    print(f"chroma_db exists: {chroma_exists} | pkls exist: {pkls_exist}")

    if chroma_exists and pkls_exist:
        print("Index files already present — skipping download.")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set — attempting anonymous download.")

    print("Downloading index files from Hugging Face Hub…")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for fname in ["bm25_index.pkl", "parents.pkl"]:
        target = INDEX_DIR / fname
        if not target.exists():
            print(f"  {fname}…")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                repo_type="dataset",
                token=token,
                local_dir=str(INDEX_DIR),
            )

    if not chroma_exists:
        print("  chroma_db/…")
        chroma_path.mkdir(exist_ok=True)
        for file_path in list_repo_files(REPO_ID, repo_type="dataset", token=token):
            if file_path.startswith("chroma_db/"):
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file_path,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(INDEX_DIR),
                )

    print("Download complete.")


if __name__ == "__main__":
    download_index()
