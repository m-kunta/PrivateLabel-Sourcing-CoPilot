# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Generate synthetic lead time data (creates data/vendor_lead_times.csv)
python data_gen.py

# Initialize Pinecone index and ingest lead times
python -c "
from vector_store import VectorStore
import os; from dotenv import load_dotenv; load_dotenv()
vs = VectorStore(os.getenv('PINECONE_API_KEY'))
vs.init_index()
vs.ingest_lead_times('data/vendor_lead_times.csv')
"

# Ingest live disruption events into Pinecone
python -c "
from vector_store import VectorStore
from rss_ingest import get_live_disruptions
import os; from dotenv import load_dotenv; load_dotenv()
vs = VectorStore(os.getenv('PINECONE_API_KEY'))
vs.ingest_disruptions(get_live_disruptions())
"

# Launch dashboard
streamlit run app.py
```

## Architecture

PL Sourcing Co-Pilot is a private label supply chain intelligence platform. It answers "what-if" disruption questions using Retrieval-Augmented Generation (RAG) over a Pinecone vector database of vendor lead times and disruption news.

### Data Flow

```
data_gen.py → data/vendor_lead_times.csv
                        ↓
              VectorStore.ingest_lead_times()
                        ↓
              Pinecone (lead_times namespace)
                        ↓
rss_ingest.py → disruption events
                        ↓
              VectorStore.ingest_disruptions()
                        ↓
              Pinecone (disruptions namespace)
                        ↓
User question → StrategicAnalystChain.analyze_scenario()
                  ├── VectorStore.query(lead_times, top_k=15)
                  ├── VectorStore.query(disruptions, top_k=5)
                  ├── Python computes risk_table from vector metadata + heuristic coefficients
                  └── LLM(briefing_system_prompt, risk_table + disruption context)
                        ↓ generates ripple_effects + briefing only
              Merged JSON → Streamlit UI
```

### Module Responsibilities

| Module | Class/Functions | Role |
|--------|----------------|------|
| `data_gen.py` | standalone script | Generates 50-row synthetic lead time CSV |
| `vector_store.py` | `VectorStore` | Pinecone init, embed, ingest, query |
| `rss_ingest.py` | `get_live_disruptions()` | Fetches from live RSS feeds and parses via LLM |
| `llm_providers.py` | `get_llm_response()` | Multi-provider LLM factory (lazy imports) |
| `disruption_model.py` | `classify_scenario()`, `build_risk_row()` | Shared disruption classification, coefficient selection, exposure matching |
| `scenario_engine.py` | `StrategicAnalystChain` | RAG orchestration + LLM reasoning |
| `app.py` | Streamlit app | 3-tab UI: Scenario Analyzer, Risk Dashboard, Data Hub |

### Pinecone Setup

- **Index name:** `pl-sourcing-copilot` (configurable via `PINECONE_INDEX_NAME` env var)
- **Dimension:** 384 (sentence-transformers `all-MiniLM-L6-v2`)
- **Metric:** cosine
- **Cloud:** AWS us-east-1 (serverless)
- **Namespaces:** `lead_times`, `disruptions`

Pinecone v3 API imports:
```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.create_index(name=..., dimension=384, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(index_name)
```

### Embedding Pattern

The embedder is lazy-loaded on first use to avoid slow startup:
```python
# In VectorStore.__init__:
self._embedder = None

def _get_embedder(self):
    if self._embedder is None:
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return self._embedder

def embed(self, text: str) -> list[float]:
    return self._get_embedder().encode(text).tolist()
```

### LLM Provider Pattern

Follows the same lazy-import factory pattern as `dc_outbound_smoothing/llm_providers.py`:

```python
# Extended signature with system_prompt support and max_tokens=2000
def get_llm_response(prompt: str, provider: str, model: str,
                     system_prompt: str = None) -> str:
    ...
```

Provider defaults:
- Gemini: `gemini-2.5-flash`
- OpenAI: `gpt-4o-mini`
- Anthropic: `Codex-sonnet-4-20250514`
- Groq: `llama-3.3-70b-versatile`
- Ollama: `llama3.2`

### Scenario Engine Output Schema

`analyze_scenario()` returns a dict with this structure:

```python
{
    "risk_table": [
        {
            "vendor": str,
            "component": str,
            "category": str,
            "origin": str,
            "base_lead_days": int,
            "disruption_coefficient": float,  # e.g. 1.35
            "adjusted_lead_days": int,
            "risk_level": "Red" | "Yellow" | "Green",
            "risk_rationale": str
        }
    ],
    "ripple_effects": [
        {
            "primary_disruption": str,
            "affected_route": str,
            "downstream_impacts": [str]  # list of downstream consequences
        }
    ],
    "briefing": {
        "executive_summary": str,
        "key_findings": [str],
        "affected_categories": [str],
        "recommended_actions": [str],
        "risk_horizon": str  # e.g. "4–8 weeks"
    },
    "source": "rag+llm" | "rag+llm-degraded" | "fallback"
}
```

### Disruption Coefficients (Heuristic Baseline)

| Disruption | Coefficient | Notes |
|-----------|-------------|-------|
| Panama Canal -30% transit | 1.35 | Asia-East Coast; queue buildup |
| Panama Canal -50% transit | 1.55 | Severe; partial Suez reroute |
| Port of Savannah work action | 1.50 | All East Coast imports |
| Suez Canal / Red Sea closure | 1.40 | Cape of Good Hope adds ~14 days |
| West Africa port congestion | 1.25 | Cocoa/palm oil lanes |
| Bangladesh flooding | 1.30 | Chittagong inland disruption |
| Strait of Hormuz disruption | 1.45 | Fuel surcharge spike and Middle East rerouting |

### Risk Level Thresholds

```python
def classify_risk(base_lead_days: int, adjusted_lead_days: int) -> str:
    ratio = adjusted_lead_days / base_lead_days
    if ratio > 1.35:
        return "Red"
    elif ratio > 1.15:
        return "Yellow"
    else:
        return "Green"
```

### Fallback Mode

If `PINECONE_API_KEY` is not set, the engine falls back to:
1. Loading `data/vendor_lead_times.csv` directly as context
2. Applying the heuristic coefficient table without vector retrieval
3. Passing the raw CSV subset + coefficients to the LLM as text context

This ensures the app is demonstrable without a Pinecone account.

## Key Implementation Patterns

### Graceful Degradation

```python
# In VectorStore
def is_ready(self) -> bool:
    if self._ready and self.index is not None:
        return True
    ...

# In app.py / scenario_engine.py
if vs.is_ready():
    context = vs.query(question, "lead_times")
else:
    context = load_csv_as_context("data/vendor_lead_times.csv")
```

### Vector Metadata for Lead Times

Each vector in the `lead_times` namespace includes:
```python
{
    "id": f"lt_{vendor_id}_{component_slug}",
    "values": embed(text),  # 384-dim
    "metadata": {
        "vendor_id": ..., "vendor_name": ..., "brand_name": ...,
        "category": ..., "component": ..., "origin_country": ...,
        "origin_port": ..., "destination_port": ...,
        "transport_mode": ..., "route_name": ...,
        "base_lead_days": int, "historical_variance_pct": float,
        "panama_canal_exposure": 0|1,
        "suez_canal_exposure": 0|1,
        "savannah_port_exposure": 0|1,
        "west_africa_port_exposure": 0|1,
        "text": str  # the string that was embedded
    }
}
```

### Vector Metadata for Disruptions

Each vector in the `disruptions` namespace:
```python
{
    "id": f"dis_{slug}_{date}",
    "values": embed(text),
    "metadata": {
        "event_type": ..., "location": ..., "severity": ...,
        "affected_routes": "route, route",  # Pinecone-compatible string
        "affected_routes_json": "[\"route\", \"route\"]",  # original list as JSON
        "date": ..., "source": ...,
        "headline": ..., "text": str
    }
}
```

### LLM System Prompt (scenario_engine.py)

The system prompt establishes the AI persona:

```
You are a Principal Supply Chain Strategist specializing in private label raw material sourcing.
You have deep expertise in global freight routes, commodity supply chains, and disruption impact analysis.

When analyzing supply chain scenarios, you must:
1. Identify the primary disruption and which shipping routes / port chokepoints are affected.
2. Apply disruption coefficients to base lead times based on historical precedent.
3. Trace the "ripple effect" — second-order impacts on other components sharing the same route.
4. Classify each at-risk component by risk level: Red (>35% lead time increase), Yellow (15–35%), Green (<15%).
5. Generate a professional briefing suitable for a VP of Merchandising.

Always respond in valid JSON matching the specified schema. Do not include markdown code fences in your response.
```

### Streamlit Session State Keys

```python
st.session_state["vs"]           # VectorStore instance
st.session_state["chain"]        # StrategicAnalystChain instance
st.session_state["lead_times_df"] # DataFrame loaded from CSV
st.session_state["last_analysis"] # Most recent analyze_scenario() result
st.session_state["disruptions"]  # List of disruption event dicts
st.session_state["provider"]     # Selected LLM provider name
st.session_state["model"]        # Selected LLM model name
```

## Data Schema: vendor_lead_times.csv

| Column | Type | Example |
|--------|------|---------|
| vendor_id | str | V001 |
| vendor_name | str | True Brand Cotton Textiles |
| brand_name | str | True |
| category | str | Apparel/Textiles |
| component | str | Cotton |
| origin_country | str | Bangladesh |
| origin_port | str | Chittagong |
| destination_port | str | Savannah |
| transport_mode | str | Ocean Freight |
| route_name | str | Asia-East Coast |
| base_lead_days | int | 45 |
| historical_variance_pct | float | 12.5 |
| panama_canal_exposure | int | 1 (boolean) |
| suez_canal_exposure | int | 0 (boolean) |
| savannah_port_exposure | int | 1 (boolean) |
| west_africa_port_exposure | int | 0 (boolean) |
| hrmz_exposure | int | 0 (boolean) |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PINECONE_API_KEY` | No | — | Pinecone API key (omit for fallback mode) |
| `PINECONE_INDEX_NAME` | No | `pl-sourcing-copilot` | Pinecone index name |
| `LLM_PROVIDER` | Yes | `Anthropic` | Active LLM provider |
| `ANTHROPIC_API_KEY` | If using Anthropic | — | Codex API key |
| `OPENAI_API_KEY` | If using OpenAI | — | OpenAI API key |
| `GEMINI_API_KEY` | If using Gemini | — | Google Gemini API key |
| `GROQ_API_KEY` | If using Groq | — | Groq API key |

## Testing Approach

### Manual Smoke Tests

```bash
# 1. Verify data generation
python data_gen.py && python -c "import pandas as pd; df = pd.read_csv('data/vendor_lead_times.csv'); print(len(df), 'rows')"
# Expected: 50 rows

# 2. Verify VectorStore is_ready() returns False without Pinecone key
python -c "from vector_store import VectorStore; vs = VectorStore(None); print(vs.is_ready())"
# Expected: False

# 3. Verify fallback analysis runs without Pinecone
python -c "
import pandas as pd
from scenario_engine import StrategicAnalystChain
chain = StrategicAnalystChain(vector_store=None, provider='Anthropic', model='Codex-3-7-sonnet-20250219')
df = pd.read_csv('data/vendor_lead_times.csv')
result = chain._fallback_analysis('Panama Canal -30% transit', df)
print(result['source'])  # Expected: 'fallback'
"

# 4. Launch app
streamlit run app.py
# Expected: loads at localhost:8501 without errors
```

### Scenario Regression Tests (`tests/test_backend.py`)

Run with:
```bash
cd pl_sourcing_copilot
source .venv/bin/activate
pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/test_backend.py -v
```

Implemented test cases:
- `test_risk_classification_thresholds()` — P0: Red/Yellow/Green boundary values ✅
- `test_disruption_coefficient_panama()` — P0: Panama -30% → coefficient 1.35, correct row selection ✅
- `test_json_schema_compliance()` — P0: LLM markdown fences are stripped; JSON is parsed correctly ✅
- `test_fallback_schema_fallback()` — P0: LLM error → degraded fallback returns valid schema ✅
- `TestFallbackDisruptionCoefficients` — P0: all 5 disruption types (suez, savannah, west_africa, hormuz, unrecognized) ✅
- `TestRippleEffectSavannah` — P1: Savannah strike surfaces ≥ 2 component categories ✅
- `TestRiskTableOrdering` — P1: risk_table sorted by adjusted_lead_days descending ✅
- `TestVectorStoreRoundtrip` — P1: ingest → query returns matching record without hitting actual Pinecone API ✅

Not yet implemented:
- None

## Important Notes

- **Never edit `data/vendor_lead_times.csv` manually** — always regenerate via `data_gen.py` to maintain consistency.
- **Pinecone free tier** supports 1 serverless index with 100K vectors — well within prototype needs (50 lead times + ~20 disruptions = 70 vectors total).
- **First run** of the app downloads the `all-MiniLM-L6-v2` model (~80MB). Subsequent runs use the cached model.
- **LLM max_tokens** is set to 8192 to accommodate the full briefing document JSON and massive context outputs.
- **JSON parsing:** The LLM is instructed not to use markdown code fences. `_parse_response()` still strips them defensively using regex: `re.search(r'\{.*\}', text, re.DOTALL)`.

---

## Claude Code Skills

When working in this project with **Claude Code**, the following skills are active via `.claude/settings.json`. Invoke them with the `Skill` tool before relevant tasks:

| Skill | Trigger |
|---|---|
| `superpowers:brainstorming` | Before any new feature or architecture change |
| `superpowers:writing-plans` | Before multi-step implementation tasks |
| `superpowers:executing-plans` | When executing a written plan |
| `superpowers:systematic-debugging` | When encountering any bug or test failure |
| `superpowers:test-driven-development` | Before writing implementation code |
| `superpowers:requesting-code-review` | Before merging or after major changes |
| `superpowers:verification-before-completion` | Before claiming work is done |
| `ralph-loop:ralph-loop` | To start an autonomous development loop (`ralph --monitor`) |
| `commit-commands:commit` | When committing changes |
| `commit-commands:commit-push-pr` | When pushing and opening a PR |
| `feature-dev:feature-dev` | When implementing a new feature end-to-end |
| `octo:debug` | Deep debugging workflows |
| `octo:tdd` | Test-driven development cycles |
| `octo:review` | Code review |
| `security-guidance:security-review` | Security audit before shipping |
| `claude-mem:make-plan` | Create a phased implementation plan |
| `claude-mem:mem-search` | Search cross-session memory for prior work |
