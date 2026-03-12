import re
import os
import threading
import numpy as np
from sentence_transformers import SentenceTransformer


def _load_pronouns(pronouns_dir: str) -> dict[str, re.Pattern]:
    # Read first-person pronoun lists from data/pronouns/<lang>.txt and compile
    # one regex pattern per language. Adding a language = dropping a new file
    patterns = {}
    if not os.path.isdir(pronouns_dir):
        return patterns

    for fname in os.listdir(pronouns_dir):
        if not fname.endswith('.txt'):
            continue
        lang  = fname[:-4]
        fpath = os.path.join(pronouns_dir, fname)
        with open(fpath, encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        if not words:
            continue
        words.sort(key=len, reverse=True)
        escaped = [re.escape(w) for w in words]
        patterns[lang] = re.compile(
            r'(?<!\w)(' + '|'.join(escaped) + r')(?!\w)',
            re.IGNORECASE | re.UNICODE
        )

    return patterns


class PsychologyEngine:
    # Singleton
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, data_dir: str | None = None):
        if hasattr(self, 'initialized'):
            return

        self.model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.model.eval()  # disable dropout, slightly faster inference

        # {lang: compiled pattern} — built from files, no hardcoded languages
        pronouns_dir = os.path.join(data_dir or '', 'pronouns')
        self._pronouns: dict[str, re.Pattern] = _load_pronouns(pronouns_dir)

        self.initialized = True

    def encode(self, texts: list[str]) -> np.ndarray:
        # Returns L2-normalised float32 vectors. batch_size=16 limits peak RAM.
        vecs = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,  # dot-product == cosine on unit vectors
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def filter_personal_focus(self, text: str, lang: str) -> list[str]:
        # Keep only sentences that contain a first-person pronoun for this language.
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        pattern   = self._pronouns.get(lang)

        if pattern is None:
            return sentences  #if unknown language — skip filtering

        personal = [s for s in sentences if pattern.search(s)]
        return personal if personal else sentences  # fallback: keep everything

    def get_top_matches(self, text: str, lang: str, tags_data: list[dict]) -> list[dict]:
        sentences = self.filter_personal_focus(text, lang)
        q_vecs    = self.encode(sentences)  # (n_sent, dim), float32, normalised

        results = []
        for tag in tags_data:
            pos_matrix = tag.get('pos_matrix')  # float16, (n_markers, dim)
            if pos_matrix is None:
                continue

            # Cast markers to float32 only for multiplication 
            sims  = q_vecs @ pos_matrix.T.astype(np.float32)  # (n_sent, n_markers)
            score = float(sims.max())

            if score >= tag.get('threshold', 0.6):
                results.append({'tag_id': tag['id'], 'score': round(score, 4)})

        return sorted(results, key=lambda x: x['score'], reverse=True)
