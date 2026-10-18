"""Run: python demo.py            (vector search only, no API key needed)
       python demo.py --llm      (adds Claude fact extraction and hybrid lookup)"""

import sys

from corpus import chunk_documents
from vector_search import VectorIndex

QUERY = "Is device A4-99y compatible with System B?"
IDENTIFIER, ABOUT = "A4-99y", "System B"


def show_vector_search(index: VectorIndex) -> None:
    print(f"Vector search: {QUERY!r}")
    for score, chunk in index.search(QUERY, k=4):
        print(f"  {score:.3f}  [{chunk.doc_id} / {chunk.section}] {chunk.text}")
    print("  -> The top hits disagree with each other, and the scores can't say which one is true.\n")


def show_facts(title: str, facts) -> None:
    print(title)
    for f in facts:
        verdict = "NO" if f.negated else "YES"
        cond = f" (if {f.condition})" if f.condition else ""
        print(f"  {verdict}{cond}: {f.quote!r}  [{f.doc_id} / {f.section}]")
    print()


def main() -> None:
    index = VectorIndex(chunk_documents())
    show_vector_search(index)
    if "--llm" not in sys.argv:
        print("Pass --llm (with ANTHROPIC_API_KEY set) to run fact extraction.")
        return

    from fact_extraction import ClaudeFactExtractor, find_facts
    from hybrid import hybrid_lookup

    extractor = ClaudeFactExtractor()
    all_facts = extractor.extract_all()
    show_facts("Full LLM extraction, exact-match lookup:", find_facts(all_facts, IDENTIFIER, ABOUT))
    show_facts("Hybrid (vector top-k -> LLM extraction):",
               hybrid_lookup(index, extractor, QUERY, IDENTIFIER, ABOUT))


if __name__ == "__main__":
    main()
