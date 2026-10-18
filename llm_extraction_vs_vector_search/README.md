# LLM fact extraction vs. vector search

A small, runnable comparison of two retrieval styles on a corpus built to
trigger vector search's known failure modes.

| File | What it shows |
|---|---|
| `corpus.py` | Documents with a negation pair ("compatible" / "never compatible"), near-identical IDs (`A4-99x` / `A4-99y`), and facts that depend on section headers. |
| `vector_search.py` | Hashed bag-of-words + char-trigram embedding with cosine top-k. It stands in for a dense embedding model and fails in the same ways. |
| `fact_extraction.py` | Claude structured-output extraction into atomic facts (`negated`, verbatim `identifiers`, `section`, `quote`), then exact-match lookup. |
| `hybrid.py` | Best-practice pipeline: vector top-k picks the sections, then Claude extracts facts from only those sections. |

## Run

```bash
pip install -r requirements.txt
python demo.py          # vector search only, offline
python demo.py --llm    # adds extraction + hybrid (needs ANTHROPIC_API_KEY)
python -m pytest -q     # offline tests with a fake client
```

Offline output:

```
Vector search: 'Is device A4-99y compatible with System B?'
  0.887  [compat-matrix / System B] Device A4-99y is never compatible with System B.
  0.879  [compat-matrix / System B] Device A4-99x is compatible with System B.
  ...
```

The contradicting sentence and the wrong device score within 0.01 of the
right answer. After extraction, the question becomes a field filter:
`identifier == "A4-99y" and object contains "System B"` returns one fact with
`negated=True`.

## Trade-offs

- **Vector search** is cheap and fast, and it scales to millions of chunks, but it
  ranks by similarity, not truth.
- **Full extraction** is precise and keeps structure, but it costs tokens on every
  document. Run it once at ingest time and store the facts. Don't run it per query.
- **Hybrid** keeps the cost bounded by k and still answers from structured facts.

Extraction uses `claude-opus-5-5` with `output_config.format` (JSON schema from
the Pydantic model) and server-side refusal fallbacks (`fallbacks="default"`).
