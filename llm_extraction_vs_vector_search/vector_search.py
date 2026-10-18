"""Minimal embedding + cosine-similarity search.

The embedding is a hashed bag of word unigrams and character trigrams. It is a
stand-in for a dense embedding model and fails in the same ways: "never" is one
token among many, and A4-99x / A4-99y share almost every trigram.
"""

import re
import zlib

import numpy as np

from corpus import Chunk

DIM = 2048


def _features(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9.-]+", text.lower())
    grams = [f"#{w[i:i + 3]}" for w in words for i in range(max(1, len(w) - 2))]
    return words + grams


def embed(text: str) -> np.ndarray:
    vec = np.zeros(DIM)
    for feature in _features(text):
        vec[zlib.crc32(feature.encode()) % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class VectorIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.matrix = np.stack([embed(c.text) for c in chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[float, Chunk]]:
        scores = self.matrix @ embed(query)
        order = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in order]
