"""LLM-based atomic fact extraction with Claude structured outputs.

Each fact keeps exact identifiers, an explicit negation flag, and an anchor back
to its source document, section and verbatim quote, so downstream code can
filter on fields instead of trusting embedding distance.
"""

import json

import anthropic
from pydantic import BaseModel, ConfigDict

from corpus import DOCUMENTS

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You extract atomic facts from technical documents.

Return one fact per standalone claim. Copy identifiers (model numbers, versions,
codes) character for character. Set "negated" to true when the claim states that
something is NOT the case ("never", "not", "unsupported", "deprecated").
"section" is the nearest heading above the claim and "quote" is the exact source
sentence. Do not infer facts that the text does not state."""


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    negated: bool
    condition: str | None
    identifiers: list[str]
    doc_id: str
    section: str
    quote: str


class FactList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[Fact]


class ClaudeFactExtractor:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = MODEL):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def extract(self, doc_id: str, text: str) -> list[Fact]:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": FactList.model_json_schema()},
            },
            # Re-run on Anthropic's recommended model if a safety classifier declines.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{
                "role": "user",
                "content": f"<document id=\"{doc_id}\">\n{text}\n</document>",
            }],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Extraction refused for {doc_id}: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"Extraction truncated for {doc_id}")
        payload = "".join(b.text for b in response.content if b.type == "text")
        return FactList.model_validate(json.loads(payload)).facts

    def extract_all(self, documents: dict[str, str] = DOCUMENTS) -> list[Fact]:
        return [fact for doc_id, text in documents.items() for fact in self.extract(doc_id, text)]


def find_facts(facts: list[Fact], identifier: str, about: str) -> list[Fact]:
    """Exact-match lookup: the identifier must appear verbatim, so A4-99x never
    answers a question about A4-99y."""
    about = about.lower()
    return [
        f for f in facts
        if identifier in f.identifiers and about in f"{f.object} {f.condition or ''}".lower()
    ]
