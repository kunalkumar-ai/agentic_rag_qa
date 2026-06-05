import pdfplumber
from config import DOCUMENTS


def extract_pdf(pdf_path: str, txt_path: str) -> None:
    """Extract text from a PDF using pdfplumber and save to a .txt file."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if text:
                pages.append(text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(pages))
    print(f"  Extracted {len(pages)} pages → {txt_path}")


def extract_all() -> None:
    """Extract all PDFs in the DOCUMENTS registry to .txt files."""
    for doc in DOCUMENTS:
        print(f"Extracting {doc['company']} {doc['year']}...")
        extract_pdf(doc["path"], doc["txt_path"])
    print("\nExtraction complete.")


if __name__ == "__main__":
    extract_all()
