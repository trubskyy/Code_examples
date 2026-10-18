import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus import chunk_documents  # noqa: E402
from fact_extraction import ClaudeFactExtractor, find_facts  # noqa: E402
from hybrid import hybrid_lookup  # noqa: E402
from vector_search import VectorIndex  # noqa: E402

CANNED = {
    "facts": [
        {"subject": "Device A4-99x", "predicate": "compatible with", "object": "System B",
         "negated": False, "condition": None, "identifiers": ["A4-99x"],
         "doc_id": "compat-matrix", "section": "System B",
         "quote": "Device A4-99x is compatible with System B."},
        {"subject": "Device A4-99y", "predicate": "compatible with", "object": "System B",
         "negated": True, "condition": None, "identifiers": ["A4-99y"],
         "doc_id": "compat-matrix", "section": "System B",
         "quote": "Device A4-99y is never compatible with System B."},
    ]
}


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            stop_details=None,
            content=[SimpleNamespace(type="text", text=json.dumps(CANNED))],
        )


def fake_extractor():
    messages = FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeFactExtractor(client=client), messages


def test_vector_search_cannot_separate_negation():
    hits = VectorIndex(chunk_documents()).search("Is device A4-99y compatible with System B?", k=2)
    texts = {c.text for _, c in hits}
    assert texts == {
        "Device A4-99y is never compatible with System B.",
        "Device A4-99x is compatible with System B.",
    }
    assert abs(hits[0][0] - hits[1][0]) < 0.05


def test_exact_lookup_uses_identifier_and_negation():
    extractor, messages = fake_extractor()
    facts = extractor.extract("compat-matrix", "...")
    [fact] = find_facts(facts, "A4-99y", "System B")
    assert fact.negated
    request = messages.calls[0]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["fallbacks"] == "default"


def test_hybrid_sends_only_matched_sections():
    extractor, messages = fake_extractor()
    index = VectorIndex(chunk_documents())
    query = "Is device A4-99y compatible with System B?"
    facts = hybrid_lookup(index, extractor, query, "A4-99y", "System B", k=2)
    assert [f.negated for f in facts] == [True]
    # Both top-2 chunks live in one section, so one focused extraction call.
    assert len(messages.calls) == 1
    sent = messages.calls[0]["messages"][0]["content"]
    assert "## System B" in sent and "## System C" not in sent
