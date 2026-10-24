"""Download public-domain books from Project Gutenberg into ./books/.

    python download_books.py              # the default set below
    python download_books.py 84 2701      # any Gutenberg ebook ids (Frankenstein, Moby-Dick)
"""

# Uses only the Python standard library (nothing to install).
# PYTHON NOTES: see the top of model.py for the basics.

import argparse            # command-line option parsing
import re                  # regular expressions
import urllib.request      # HTTP downloads
from pathlib import Path   # object-oriented file paths: Path("books") / "x.txt" -> books/x.txt

# A dict mapping Gutenberg ebook id -> file name to save it under.
# Find more ids by searching gutenberg.org; the id is the number in the book's URL.
DEFAULT_BOOKS = {
    11: "alice_in_wonderland",
    1661: "sherlock_holmes",
    1342: "pride_and_prejudice",
}

# {id} placeholders are filled in later with URL.format(id=11).
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def strip_gutenberg_boilerplate(text):
    """Keep only the book itself, between the *** START / *** END markers.

    Every Gutenberg file begins and ends with a licence and other text that
    isn't part of the book; we don't want the model to learn that.
    """
    # r"..." is a "raw string": backslashes are kept literally, which is what
    # regular expressions need. \* matches a literal *, " ?" an optional space,
    # (THE|THIS) either word, and .*? the shortest run of any characters.
    # re.search returns a match object (with .start()/.end() positions) or None.
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    # `a if cond else b` is Python's ternary operator (cond ? a : b in C).
    # Slice from just after the START marker to just before the END marker,
    # falling back to the whole text if a marker is missing.
    text = text[start.end() if start else 0 : end.start() if end else len(text)]
    # .strip() removes leading/trailing whitespace.
    return text.strip() + "\n"


def download(book_id, name, out_dir):
    """Download one book to out_dir/name.txt, unless it's already there."""
    path = out_dir / f"{name}.txt"   # the / operator joins Path objects
    if path.exists():
        print(f"already have {path}")
        return path
    print(f"downloading {URL.format(id=book_id)}")
    # Some servers reject requests without a User-Agent header, so we set one.
    req = urllib.request.Request(URL.format(id=book_id), headers={"User-Agent": "tiny-llm-example"})
    # `with` guarantees the network connection is closed afterwards.
    with urllib.request.urlopen(req, timeout=60) as resp:
        # .read() returns raw bytes; .decode() turns bytes into a text string.
        # "utf-8-sig" also strips the invisible byte-order mark some files start with.
        raw = resp.read().decode("utf-8-sig", errors="replace")
    path.write_text(strip_gutenberg_boilerplate(raw), encoding="utf-8")
    # // is integer division: bytes -> whole kilobytes.
    print(f"  saved {path} ({path.stat().st_size // 1024} KB)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # nargs="*" = zero or more positional values; type=int converts each to an integer.
    ap.add_argument("ids", nargs="*", type=int, help="Gutenberg ebook ids (default: a few classics)")
    ap.add_argument("--out", default="books", help="output folder")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)   # create the folder; no error if it already exists
    # An empty list counts as False in Python, so this uses the ids given on the
    # command line if there are any, otherwise the default books.
    books = {i: f"book_{i}" for i in args.ids} if args.ids else DEFAULT_BOOKS
    # .items() yields (key, value) pairs, unpacked into two loop variables.
    for book_id, name in books.items():
        download(book_id, name, out_dir)


# Run main() only when this file is executed directly, not when imported.
if __name__ == "__main__":
    main()
