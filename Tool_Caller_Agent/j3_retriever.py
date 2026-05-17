# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC)
#  Product Signature: FTC
# ------------------------------------------------------------------------------
#  File: Tool_Caller_Agent/j3_retriever.py
#  Purpose: TF-IDF retrieval interface over the 50-doc SOC corpus.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-05-17
#  Environment: Python 3.11
# =============================================================================

"""Minimal TF-IDF retriever for the J=3 SOC corpus.

Usage:
    from Tool_Caller_Agent.j3_retriever import Retriever
    r = Retriever()
    results = r.retrieve("lateral movement credential reset", k=3)
"""

import json
import os
import pickle
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "j3_corpus")
INDEX_CACHE_PATH = os.path.join(CORPUS_DIR, "_tfidf_index.pkl")


class Retriever:
    """TF-IDF cosine retriever over the 50-doc SOC corpus."""

    def __init__(self, corpus_dir: str = CORPUS_DIR, rebuild: bool = False) -> None:
        self.corpus_dir = corpus_dir
        self.doc_ids: List[str] = []
        self.doc_contents: List[str] = []
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = None

        if not rebuild and os.path.exists(INDEX_CACHE_PATH):
            self._load_cache()
        else:
            self._build_index()
            self._save_cache()

    def _build_index(self) -> None:
        """Load all doc_*.txt files and build TF-IDF matrix."""
        files = sorted(
            f for f in os.listdir(self.corpus_dir)
            if f.startswith("doc_") and f.endswith(".txt")
        )
        if not files:
            raise FileNotFoundError(
                f"No doc_*.txt files found in {self.corpus_dir}"
            )

        self.doc_ids = []
        self.doc_contents = []
        for fname in files:
            fpath = os.path.join(self.corpus_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            doc_id = fname.replace(".txt", "")
            self.doc_ids.append(doc_id)
            self.doc_contents.append(content)

        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_contents)

    def _save_cache(self) -> None:
        """Persist the index to disk."""
        cache = {
            "doc_ids": self.doc_ids,
            "doc_contents": self.doc_contents,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
        }
        with open(INDEX_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    def _load_cache(self) -> None:
        """Load the cached index."""
        with open(INDEX_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        self.doc_ids = cache["doc_ids"]
        self.doc_contents = cache["doc_contents"]
        self.vectorizer = cache["vectorizer"]
        self.tfidf_matrix = cache["tfidf_matrix"]

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Return top-k documents by TF-IDF cosine similarity.

        Returns list of dicts with keys: doc_id, score, content.
        """
        if self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        if k >= len(scores):
            top_indices = scores.argsort()[::-1]
        else:
            top_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append({
                "doc_id": self.doc_ids[idx],
                "score": round(score, 6),
                "content": self.doc_contents[idx],
            })
        return results


if __name__ == "__main__":
    r = Retriever(rebuild=True)
    print(f"Indexed {len(r.doc_ids)} documents")
    test_results = r.retrieve("lateral movement credential reset", k=3)
    print(f"\nTest query: 'lateral movement credential reset'")
    for res in test_results:
        print(f"  {res['doc_id']}: score={res['score']:.4f}, "
              f"content_len={len(res['content'])}")
