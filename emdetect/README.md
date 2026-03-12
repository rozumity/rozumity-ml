# EmDetect v2.1

Multilingual semantic analyser for emotional and psychological state detection. Uses sentence embeddings to match free-form text against curated marker phrases across six psychological categories.

## How it works

Input text is split into sentences. Sentences containing first-person pronouns are selected and encoded into vectors using `paraphrase-multilingual-MiniLM-L12-v2`. These vectors are compared against pre-encoded marker matrices via dot-product similarity. Categories that exceed their configured threshold are returned as results.

Marker embeddings are cached to disk on first run — subsequent startups are fast.

## Languages

| Code | Language   | Pronoun filter |
|------|------------|----------------|
| `ru` | Russian    | regex          |
| `uk` | Ukrainian  | regex          |
| `en` | English    | regex          |
| `de` | German     | regex          |

Language is auto-detected via `langdetect` if not specified in the request.

## Categories

| ID                    | Threshold |
|-----------------------|-----------|
| `anxiety_focus`       | 0.65      |
| `burnout_focus`       | 0.75      |
| `depressive_mood`     | 0.80      |
| `aggression_irritation` | 0.75    |
| `somatic_distress`    | 0.55      |
| `low_self_esteem`     | 0.70      |

Thresholds are configurable in `src/config.json`.

## Project structure

```
emdetect_v2/
├── Dockerfile
├── .env.example
├── src/
│   ├── server.py          # FastAPI application
│   ├── main.py            # CLI entry point
│   ├── config.json        # Category definitions and thresholds
│   ├── requirements.txt
│   ├── run.sh             # Local dev runner
│   ├── ml/
│   │   ├── engine.py      # Embedding model, filtering, matching
│   │   └── loader.py      # Marker loading and cache management
│   └── data/
│       ├── markers/       # Phrase lists per category (.txt)
│       ├── pronouns/      # First-person pronoun lists per language (.txt)
│       └── cache/         # Auto-generated embedding cache (pickle)
```

## Running with Docker

```bash
# Build
docker build -t emdetect .

# Run
docker run -d \
  -p 8000:8000 \
  -e EMDETECT_API_KEY=your-secret-key \
  -v hf_cache:/app/.cache/huggingface \
  emdetect
```

The HuggingFace model (~120MB) is downloaded on first startup and cached in the volume. Subsequent restarts are instant.

## Running locally

```bash
pip install -r src/requirements.txt
export EMDETECT_API_KEY=your-secret-key
bash src/run.sh
```

If `EMDETECT_API_KEY` is not set, a random key is generated and printed to stdout on startup.

## API

### `POST /analyze`

Requires `X-API-Key` header.

**Request:**
```json
{
  "text": "I feel completely exhausted and empty inside",
  "lang": "en"
}
```

`lang` is optional — omit to auto-detect.

**Response:**
```json
{
  "lang": "en",
  "detected": true,
  "results": [
    { "category": "burnout_focus",   "score": 0.812, "level": "medium" },
    { "category": "depressive_mood", "score": 0.763, "level": "medium" }
  ]
}
```

Score levels: `low` (≤0.70) · `medium` (0.70–0.85) · `high` (>0.85)

### `GET /health`

No auth required.

```json
{ "status": "ok", "ready": true, "categories": 6, "uptime_sec": 120.4 }
```

## Extending

**Adding a category:** create a `.txt` file with marker phrases, add an entry to `config.json`, delete the corresponding cache file.

**Adding a language:** drop a `<lang>.txt` pronoun file into `src/data/pronouns/`. No code changes needed.
