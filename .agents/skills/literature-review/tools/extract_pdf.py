#!/usr/bin/env python3
"""Extract text from PDF files. Works on Windows and Linux.

Usage:
    python extract_pdf.py <input.pdf> [output.txt]

If output.txt is omitted, prints to stdout.
Requires: pip install pymupdf
"""

import sys
import pathlib


def extract_text(pdf_path: str) -> str:
    try:
        import fitz  # pymupdf
    except ImportError:
        sys.exit("Missing dependency. Install with: pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input.pdf> [output.txt]", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    if not pathlib.Path(pdf_path).exists():
        sys.exit(f"File not found: {pdf_path}")

    text = extract_text(pdf_path)

    if output_path:
        pathlib.Path(output_path).write_text(text, encoding="utf-8")
        print(f"Extracted to {output_path}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
