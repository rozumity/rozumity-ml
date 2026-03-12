import json
import os
import hashlib
import pickle
import logging
import numpy as np
from ml.engine import PsychologyEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _txt_hash(path: str) -> str:
    # MD5 of the marker file 
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _cache_path(txt_path: str) -> str:
    base = os.path.dirname(os.path.dirname(txt_path))
    name = os.path.splitext(os.path.basename(txt_path))[0]
    return os.path.join(base, 'cache', f'{name}.cache.pkl')


class MarkLoader:
    def __init__(self, engine: PsychologyEngine):
        self.engine = engine

    def load_tags_from_config(self, config_path: str) -> list[dict]:
        base_dir = os.path.dirname(os.path.abspath(config_path))

        if not os.path.exists(config_path):
            logger.error(f"Config not found: {config_path}")
            return []

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read config: {e}")
            return []

        cache_dir = os.path.join(base_dir, 'data', 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        tags_data = []

        for item in config_data:
            tag_id    = item.get('id')
            threshold = item.get('threshold', 0.6)
            full_path = os.path.join(base_dir, item.get('file_path'))

            if not os.path.exists(full_path):
                logger.error(f"Marker file not found: {full_path}")
                continue

            with open(full_path, 'r', encoding='utf-8') as f:
                phrases = [line.strip() for line in f if line.strip()]

            if not phrases:
                logger.warning(f"Marker file is empty: {full_path}")
                continue

            cache      = _cache_path(full_path)
            cur_hash   = _txt_hash(full_path)
            pos_matrix = None  # float16 ndarray after loading

            # Try loading cached embeddings
            if os.path.exists(cache):
                try:
                    with open(cache, 'rb') as f:
                        cached = pickle.load(f)
                    if cached.get('hash') == cur_hash:
                        pos_matrix = cached['matrix']  # already float16
                        logger.info(f"Cache hit: '{tag_id}' ({pos_matrix.shape[0]} phrases)")
                except Exception:
                    logger.warning(f"Cache corrupted, re-encoding: '{tag_id}'")

            # Cache miss — encode and save
            if pos_matrix is None:
                logger.info(f"Encoding '{tag_id}': {len(phrases)} phrases...")
                matrix_f32 = self.engine.encode(phrases)        # float32, normalised
                pos_matrix = matrix_f32.astype(np.float16)      # halve memory footprint
                try:
                    with open(cache, 'wb') as f:
                        pickle.dump({'hash': cur_hash, 'matrix': pos_matrix}, f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    logger.warning(f"Could not save cache: {e}")

            tags_data.append({
                'id':         tag_id,
                'threshold':  threshold,
                'pos_matrix': pos_matrix,  # float16, shape (n, dim)
            })

        logger.info(f"Loaded {len(tags_data)} tags")
        return tags_data
