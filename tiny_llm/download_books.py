"""Download public-domain books from Project Gutenberg into ./books/.

    python download_books.py              # the default set below
    python download_books.py 84 2701      # any Gutenberg ebook ids (Frankenstein, Moby-Dick)
"""

import argparse
import re
import urllib.request
from pathlib import Path

DEFAULT_BOOKS = {
    11: "alice_in_wonderland",
    1661: "sherlock_holmes",
    1342: "pride_and_prejudice",
}

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def strip_gutenberg_boilerplate(text):
    """Keep only the book itself, between the *** START / *** END markers."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    text = text[start.end() if start else 0 : end.start() if end else len(text)]
    return text.strip() + "\n"


def download(book_id, name, out_dir):
    path = out_dir / f"{name}.txt"
    if path.exists():
        print(f"already have {path}")
        return path
    print(f"downloading {URL.format(id=book_id)}")
    req = urllib.request.Request(URL.format(id=book_id), headers={"User-Agent": "tiny-llm-example"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8-sig", errors="replace")
    path.write_text(strip_gutenberg_boilerplate(raw), encoding="utf-8")
    print(f"  saved {path} ({path.stat().st_size // 1024} KB)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", type=int, help="Gutenberg ebook ids (default: a few classics)")
    ap.add_argument("--out", default="books", help="output folder")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    books = {i: f"book_{i}" for i in args.ids} if args.ids else DEFAULT_BOOKS
    for book_id, name in books.items():
        download(book_id, name, out_dir)


if __name__ == "__main__":
    main()
