"""Hybrid retrieval: vector search for cheap rough filtering, then focused LLM
fact extraction on only the top-k chunks' source sections."""

from corpus import DOCUMENTS
from fact_extraction import ClaudeFactExtractor, Fact, find_facts
from vector_search import VectorIndex


def hybrid_lookup(
    index: VectorIndex,
    extractor: ClaudeFactExtractor,
    query: str,
    identifier: str,
    about: str,
    k: int = 4,
    documents: dict[str, str] = DOCUMENTS,
) -> list[Fact]:
    hits = index.search(query, k=k)
    # Send whole sections rather than bare chunks so headers and neighbouring
    # sentences survive; this is the context that chunking threw away.
    sections = {(c.doc_id, c.section) for _, c in hits}
    facts = []
    for doc_id, section in sorted(sections):
        body = documents[doc_id]
        start = body.index(f"## {section}")
        end = body.find("\n## ", start + 1)
        facts += extractor.extract(doc_id, body[start:end if end != -1 else None])
    return find_facts(facts, identifier, about)
