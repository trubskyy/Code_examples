"""A tiny corpus built to trigger the classic vector-search failure modes.

- Negation: two sentences that differ only by "never".
- Near-identical identifiers: A4-99x vs A4-99y.
- Structure: facts that only make sense under their section header.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    section: str
    text: str


DOCUMENTS = {
    "compat-matrix": """# Compatibility Matrix

## System B
Device A4-99x is compatible with System B.
Device A4-99y is never compatible with System B.

## System C
Device A4-99y is compatible with System C after firmware 2.3.1.
""",
    "release-notes": """# Release Notes

## Firmware 2.3.1
Adds support for System C on device A4-99y.
Fixes a power-management bug on device A4-99x.

## Firmware 2.2.0
Deprecated: legacy pairing mode for System B.
""",
}


def chunk_documents(documents: dict[str, str] = DOCUMENTS) -> list[Chunk]:
    """Naive line-level chunking: one chunk per sentence, section header dropped
    from the text (as most chunkers do) but kept as metadata for display."""
    chunks = []
    for doc_id, body in documents.items():
        section = ""
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("## "):
                section = line[3:]
            elif line and not line.startswith("#"):
                chunks.append(Chunk(doc_id, section, line))
    return chunks
