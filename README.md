# PL Sourcing Co-Pilot

**A Private Label Supply Chain Intelligence Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-00B388?logo=pinecone&logoColor=white)](https://pinecone.io)
[![LLM: Multi-Provider](https://img.shields.io/badge/LLM-Claude%20%7C%20GPT--4%20%7C%20Gemini%20%7C%20Groq-8A2BE2)](https://github.com/m-kunta/PrivateLabel-Sourcing-CoPilot)
[![RAG Powered](https://img.shields.io/badge/RAG-Pinecone%20%2B%20MiniLM-0099FF)](https://github.com/m-kunta/PrivateLabel-Sourcing-CoPilot)
[![Supply Chain AI](https://img.shields.io/badge/domain-Supply%20Chain%20AI-orange)](https://github.com/m-kunta/PrivateLabel-Sourcing-CoPilot)
[![GitHub](https://img.shields.io/badge/author-m--kunta-181717?logo=github)](https://github.com/m-kunta)

> "If the Panama Canal transit capacity drops by 30%, how does that ripple through our cocoa butter and furniture component lead times?"

**PL Sourcing Co-Pilot** is a RAG-powered AI prototype that answers strategic supply chain "what-if" questions for private label buyers. It combines a Pinecone vector knowledge base (vendor lead times + live disruption news) with an LLM reasoning chain to generate risk-ranked component tables and professional analyst briefings — in seconds.

![Scenario Analyzer UI](assets/screenshot_scenario_analyzer.png)

> **✅ Pinecone Active** | **🧠 Multi-LLM** | **📊 Risk Dashboard** | **🔴 Hormuz Toggle**

---

## How to Use

### The Problem in One Sentence

Private label buying teams have **no rapid way to answer "what-if" disruption questions** — _"If Savannah strikes next week, which of our 40+ sourced components are at risk, and by how many days?"_ — before shelf gaps materialise and emergency air freight bills arrive.

### How This Solution Demonstrates the Answer

The Co-Pilot closes that gap in three steps:

1. **Stores institutional knowledge** — vendor lead times, shipping routes, and route-vulnerability flags — in a Pinecone vector database so the system understands _why_ a Panama Canal drought affects Asia-East Coast apparel but not West Africa cocoa.
2. **Reasons over a disruption scenario** — an LLM acting as a Principal Supply Chain Strategist applies heuristic disruption coefficients, traces second-order ripple effects, and risk-classifies every affected component (🔴 Red / 🟡 Yellow / 🟢 Green).
3. **Delivers a decision-ready briefing** — executive summary, key findings, recommended actions, and a risk horizon — in seconds, formatted for a VP of Merchandising.

### Quick-Start Steps

> **No Pinecone account?** Skip steps 3–5 — the app runs in **fallback mode** using `data/vendor_lead_times.csv` directly. Results are structurally identical.

**1. Install**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure**
```bash
cp .env.example .env
# Set LLM_PROVIDER and the matching API key.
# Optionally set PINECONE_API_KEY for full RAG mode.
```

**3. Generate lead time data**
```bash
python data_gen.py          # creates data/vendor_lead_times.csv
```

**4. Launch the dashboard**
```bash
streamlit run app.py        # opens at http://localhost:8501
```

**5. (Full RAG mode only) Ingest data**

In the **Data Hub** tab: click **Initialize Pinecone Index → Ingest Lead Times → Ingest Disruptions**.

**6. Run a scenario**

Go to the **Scenario Analyzer** tab and try one of these:
- *"Panama Canal transit drops 30% — which furniture components are at risk?"*
- *"Savannah port strike declared — which private label components are affected?"*
- Use the **🔴 Hormuz Toggle** to model the 2026 Strait of Hormuz geopolitical scenario instantly.

The output is a ranked risk table + an analyst briefing you can share directly with your buying team.

---

## The Problem

Private label teams at large retailers (Target, Costco, Walmart, Kroger) source raw materials directly from global suppliers under their own brand labels — True brand cotton, Frederick's cocoa butter, Casa Home wood pulp. These buyers face a structurally harder sourcing problem than branded goods buyers:

1. **No buffer from brand-name intermediaries.** When a Port of Savannah strike delays a container ship, the private label cocoa butter buyer has no Nestlé procurement team shielding them. They absorb the delay directly.

2. **Multi-tier opacity.** A cocoa butter shipment from Tema (Ghana) → Savannah → Baltimore touches three ocean freight legs, two consolidation points, and a trucking move. Each node can fail independently.

3. **Asymmetric information.** Disruption news (Red Sea shipping reroutes, Panama Canal draft restrictions, West Africa cocoa harvest failure) is public but fragmented across freight news outlets, government advisories, and commodity reports. No private label team has the bandwidth to monitor and synthesize all of it.

4. **Ripple-effect blindness.** A Savannah port slowdown doesn't just delay cotton — it delays the *spandex fiber*, the *thread*, and the *elastic trim* that all travel on the same East Coast lanes, affecting finished apparel in ways that aren't visible until shelf gaps appear.

5. **Static lead time data.** Most buying teams work from spreadsheets with "standard lead times" that were last updated years ago and don't account for current route conditions.

### The Cost

- **Out-of-Stock Penalties:** Retail private label OOS (Out-of-Shelf) costs are estimated at 4–8% of annual category revenue.
- **Emergency Air Freight:** Expedited air freight to cover a missed ocean shipment costs 6–10× the ocean rate.
- **Markdown Risk:** Late raw material arrivals compress production schedules, forcing buyers into off-peak manufacturing slots at premium costs.
- **Brand Credibility:** Private label is a loyalty driver. A consumer who can't find "True" brand organic cotton tees switches to a national brand — and may not come back.

---

## The Approach

PL Sourcing Co-Pilot addresses this problem with a three-layer architecture:

### Layer 1: Vector Knowledge Base (Pinecone RAG)

Two namespaces in a Pinecone serverless index store the "institutional memory" of the sourcing operation:

**`lead_times` namespace** — Historical vendor lead time records, semantically indexed:
- Vendor + brand name, component, origin country/port, destination port
- Transport mode, shipping route, base lead days, historical variance %
- Route vulnerability flags (Panama Canal, Suez Canal, Savannah port, West Africa port exposure)

**`disruptions` namespace** — Live and curated disruption intelligence:
- Parsed from supply chain RSS feeds (Reuters, FreightWaves, maritime advisories)
- Pre-loaded with 6 curated mock events for demo/offline use
- Each event tagged with: event type, affected routes, severity (High/Medium/Low), date

When a user asks a question, the engine retrieves the *semantically most relevant* lead time records and disruption events — not a keyword match, but a contextual understanding of what "Panama Canal" means for "Asia-East Coast furniture components."

### Layer 2: Scenario Reasoning Engine (`StrategicAnalystChain`)

The retrieved context is passed to an LLM (configurable: Claude, GPT-4, Gemini, Groq, or local Ollama) acting as a **Principal Supply Chain Strategist**. The LLM:

1. **Identifies** the primary disruption type and which shipping routes it affects.
2. **Applies a Disruption Coefficient** to base lead times (heuristically grounded):
   - Panama Canal transit drop → +15–40% on Asia-East Coast routes
   - Port of Savannah work action → +30–60% on all East Coast East imports
   - Suez Canal disruption → +20–45% on Europe/Asia routes (adds ~14 days for Cape of Good Hope reroute)
   - West Africa port congestion → +15–30% on cocoa/palm oil lanes
3. **Traces the Ripple Effect** — e.g., a Savannah strike doesn't just hit cotton; it hits spandex fiber, recycled PET packaging, and palm oil that all clear the same port.
4. **Risk-Classifies** each component (Red / Yellow / Green) based on adjusted lead time vs. baseline.
5. **Generates a professional briefing** with executive summary, key findings, and recommended actions.

The engine is designed to **gracefully degrade**: if Pinecone is not configured, it applies the heuristic coefficient table directly to the CSV data without vector search.

### Layer 3: Streamlit Decision-Support UI

Three functional tabs:

| Tab | Purpose |
|-----|---------|
| **Scenario Analyzer** | Enter a "what-if" question, set disruption magnitude, get risk table + briefing. Features the **"Hormuz" Event Toggle** for 2026 geopolitical fuel/routing modeling. |
| **Risk Dashboard** | Visual overview of all components by risk exposure and lead time |
| **Data Hub** | Ingest/refresh lead time CSV and disruption news into Pinecone |

---

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)              │
│  ┌─────────────────┐ ┌──────────────┐ ┌─────────────┐   │
│  │ Scenario        │ │ Risk         │ │ Data Hub    │   │
│  │ Analyzer        │ │ Dashboard    │ │ (Ingest)    │   │
│  └────────┬────────┘ └──────────────┘ └──────┬──────┘   │
└───────────┼──────────────────────────────────┼──────────┘
            │                                  │
            ▼                                  ▼
┌───────────────────────┐         ┌────────────────────────┐
│  StrategicAnalystChain│         │  VectorStore           │
│  (scenario_engine.py) │◄───────►│  (vector_store.py)     │
│                       │         │                        │
│  1. retrieve_context()│         │  Namespace: lead_times │
│  2. generate_analysis()         │  Namespace: disruptions│
│  3. parse_response()  │         │                        │
│  4. fallback_analysis()         │  Embedding: MiniLM-L6  │
└───────────┬───────────┘         └─────────────┬──────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐         ┌────────────────────────┐
│  llm_providers.py     │         │  Pinecone Serverless   │
│                       │         │  (us-east-1)           │
│  ├── Anthropic Claude │         │                        │
│  ├── OpenAI GPT-4     │         │  Index: pl-sourcing    │
│  ├── Google Gemini    │         │  Dim: 384              │
│  ├── Groq             │         │  Metric: cosine        │
│  └── Ollama (local)   │         └────────────────────────┘
└───────────────────────┘
                                  ┌────────────────────────┐
┌───────────────────────┐         │  rss_ingest.py         │
│  data_gen.py          │         │                        │
│  → data/vendor_       │         │  ├── Reuters RSS       │
│    lead_times.csv     │         │  ├── FreightWaves RSS  │
│  (50 synthetic rows)  │         │  └── Mock Events (10)  │
└───────────────────────┘         └────────────────────────┘
```

### Data Flow

```
data_gen.py ──────────────────────┐
                                  ▼
                     data/vendor_lead_times.csv
                                  │
                     VectorStore.ingest_lead_times()
                                  │
                          Pinecone (lead_times ns)
                                  │
rss_ingest.py ────────────────────┤
   ├── RSS feeds                  │
   └── Mock disruption events     │
                     VectorStore.ingest_disruptions()
                                  │
                          Pinecone (disruptions ns)
                                  │
User question ─► StrategicAnalystChain.analyze_scenario()
                     ├── VectorStore.query(lead_times)
                     ├── VectorStore.query(disruptions)
                     └── LLM reasoning
                                  │
                 ┌────────────────┴──────────────────┐
                 ▼                ▼                   ▼
           Risk Table    Ripple Effects        Briefing Doc
```

---

## Key Concepts

### Disruption Coefficient

The disruption coefficient (`d`) is a multiplier applied to base lead times to estimate the adjusted lead time under a disruption scenario:

```
adjusted_lead_days = base_lead_days × d(disruption_type, magnitude_pct)
```

Heuristic baseline coefficients (grounded in historical freight data):

| Disruption Type | Coefficient | Basis |
|-----------------|-------------|-------|
| Panama Canal capacity -30% | 1.35 | Transit queue buildup; reroute adds 7–10 days |
| Panama Canal capacity -50% | 1.55 | Severe queuing; partial Suez reroute |
| Port of Savannah work action | 1.50 | Historical ILWU/ILA work-to-rule adds 15–20 days |
| Suez Canal closure (Red Sea) | 1.40 | Cape of Good Hope detour adds ~14 days |
| West Africa port congestion | 1.25 | Tema/Abidjan congestion typical +5–8 days |
| Bangladesh flooding | 1.30 | Chittagong inland road disruption +7–12 days |
| Strait of Hormuz Blockage | 1.45 | Massive fuel surcharge spike (+15-30% costs) and extensive Middle East rerouting |

The LLM reasons about the coefficient contextually — e.g., a Panama Canal disruption has a higher coefficient for furniture (heavy, slow transit, bulk cargo) than for textiles (can reroute via air at marginal cost).

### Ripple Effect

A single port disruption rarely affects a single component. The engine maps downstream impacts:

```
Savannah Port Strike
  ├── Cotton (True brand) ──► Delayed → apparel production schedule at risk
  ├── Spandex Fiber ──────────► Delayed → athletic wear line impacted
  ├── Recycled PET ───────────► Delayed → packaging for all private label lines
  └── Palm Oil ──────────────► Delayed → personal care manufacturing delay
```

The LLM is explicitly prompted to enumerate second-order effects when generating its briefing.

### Risk Zones

| Zone | Condition | Action |
|------|-----------|--------|
| 🔴 Red | adjusted_lead_days > base × 1.35 | Alert buyer; activate dual-source or air freight contingency |
| 🟡 Yellow | adjusted_lead_days > base × 1.15 | Monitor closely; pre-position safety stock |
| 🟢 Green | adjusted_lead_days ≤ base × 1.15 | Normal operations |

---

## File Reference

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard — 3 tabs, sidebar, session state |
| `vector_store.py` | `VectorStore` class — Pinecone v3 init, ingest, query |
| `scenario_engine.py` | `StrategicAnalystChain` — RAG orchestration + LLM reasoning |
| `rss_ingest.py` | Live RSS feed parser (Supply Chain Dive, Logistics Mgmt) + curated mock fallback |
| `llm_providers.py` | Multi-provider LLM factory (Anthropic/OpenAI/Gemini/Groq/Ollama) |
| `data_gen.py` | Generates `data/vendor_lead_times.csv` (50 synthetic rows) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template — copy to `.env` and fill in keys |
| `data/vendor_lead_times.csv` | Generated by `data_gen.py` — DO NOT edit manually |

---

## Synthetic Data Coverage

`data_gen.py` generates 50 vendor-component records across:

### Categories & Components

| Category | Components | Example Vendors |
|----------|-----------|-----------------|
| Apparel/Textiles | Cotton, Spandex Fiber, Polyester Yarn, Denim Fabric | True Brand Textiles |
| Confectionery/Food | Cocoa Butter, Cocoa Powder, Palm Oil, Vanilla Extract | Frederick's Ingredients |
| Wood/Furniture | Wood Pulp, Hardwood Lumber, MDF, Veneer | Casa Home Sourcing |
| Personal Care | Shea Butter, Beeswax, Essential Oils, Aloe Vera | NatureBest |
| Packaging | Recycled Cardboard, Glass Cullet, PET Flakes | EcoSource Pack |

### Origins & Routes

| Route | Origins | Destination | Key Vulnerabilities |
|-------|---------|-------------|---------------------|
| Trans-Pacific | China, Vietnam, Indonesia | Los Angeles, Seattle | Direct, less Panama exposure |
| Asia–East Coast (via Panama) | Bangladesh, India, China | Savannah, New York | Panama Canal, Savannah |
| West Africa–Atlantic | Ghana, Ivory Coast, Nigeria | New York, Baltimore | West Africa ports |
| South America–Atlantic | Brazil, Colombia | Miami, New Orleans | Savannah (transshipment) |
| Europe–East Coast | Sweden, Finland, Germany | Baltimore, New York | Suez (origin legs) |

---

## Setup

### Prerequisites

- Python 3.9+
- Pinecone account (free tier supports this project): [pinecone.io](https://www.pinecone.io)
- One LLM API key: Anthropic, OpenAI, Gemini, or Groq (Ollama works with no key)

### Installation

```bash
cd pl_sourcing_copilot

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required in `.env`:**
```env
PINECONE_API_KEY=your_pinecone_key_here
LLM_PROVIDER=Anthropic          # or OpenAI, Gemini, Groq, Ollama
ANTHROPIC_API_KEY=your_api_key_here
```

**Optional Alternative LLM Keys (depending on your `LLM_PROVIDER` choice):**
```env
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

### Supported LLM Providers

| Provider | Python Package | Cost | Default Model |
|---|---|---|---|
| **Anthropic** | `anthropic` | Paid | `claude-3-7-sonnet-20250219` |
| **OpenAI** | `openai` | Paid | `gpt-4o-mini` |
| **Gemini** | `google-genai` | Free Tier | `gemini-2.5-flash` |
| **Groq** | `groq` | Free API | `llama-3.3-70b-versatile` |
| **Ollama** | `ollama` | Free (Local) | `llama3.2` |

### Data Initialization

```bash
# Step 1: Generate synthetic lead time data
python data_gen.py
# → Creates data/vendor_lead_times.csv (50 rows)

# Step 2: Launch the dashboard
streamlit run app.py
```

**Step 3: Ingest Data via UI**
1. Open the app in your browser (usually `http://localhost:8501`)
2. Navigate to the **Data Hub** tab.
3. Click the **"Initialize Pinecone Index"** button.
4. Click **"Ingest Lead Times"** and **"Ingest Disruptions"** to load the data into the vector database.

> *Alternatively, you can run the initialization scripts via the CLI if you prefer, but the UI is the recommended path for first-time setup.*

---

## Demo Without Pinecone

The app works in **fallback mode** without a Pinecone API key — it applies the heuristic disruption coefficient table directly to `data/vendor_lead_times.csv`. Scenario results will be accurate in structure but won't benefit from semantic retrieval or live disruption context.

To run in fallback mode: just omit `PINECONE_API_KEY` from `.env`.

---

## Example Scenarios

```
"If the Panama Canal transit drops by 30%, how are furniture component lead times affected?"

"A port strike is declared at Savannah. Which of our private label components are at risk?"

"West Africa cocoa yield has dropped 25% due to El Niño. How does this affect cocoa butter supply?"

"If Red Sea shipping is suspended and Suez Canal routes are impacted, which routes are forced
 around the Cape of Good Hope and by how many days?"

"Which of our apparel components have the highest combined risk from Panama Canal + Savannah exposure?"

"With the 2026 tensions escalating, what is the fuel surcharge and transit delay impact on our Asian imports if the Strait of Hormuz is blocked?"
```

---

## Development Roadmap

- [x] Architecture design
- [x] Problem statement and documentation
- [x] `data_gen.py` — synthetic lead time CSV
- [x] `vector_store.py` — Pinecone v3 integration
- [x] `rss_ingest.py` — Live RSS feed parsing via LLM extraction (with mock disruption fallback)
- [x] `llm_providers.py` — multi-provider LLM factory
- [x] `scenario_engine.py` — RAG + LLM reasoning chain (two-step: Python risk table + LLM briefing)
- [x] `app.py` — Streamlit dashboard (3 tabs)
- [x] The "Hormuz" Event Toggle (2026 Geopolitical scenario modeling)
- [x] Unit + regression tests (`tests/test_backend.py`, `tests/test_vector_store.py`, `tests/test_scenario_engine_rag.py`, `tests/test_support_modules.py`) — 34 tests passing
- [ ] News API integration (upgrade from RSS scraping)
- [ ] Real-time Pinecone update pipeline

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Vector DB | Pinecone (serverless) | Managed, no infra, free tier covers prototype scale |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, local, no API key required |
| LLM | Configurable (Claude/GPT-4/Gemini/Groq/Ollama) | Provider-agnostic; use whatever key you have |
| UI | Streamlit | Consistent with sibling projects; rapid prototyping |
| News ingestion | `feedparser` + curated mock events | RSS is free; mock events ensure demo stability |
| Data | Pandas + SQLite (future) | Same stack as `dc_outbound_smoothing` |

---

## Risk Dashboard

![Risk Dashboard](assets/screenshot_risk_dashboard.png)

---

## Changelog

### v0.3.0 — 2026-04-26 (Live RSS & Testing)

#### Features
- **Live RSS Ingestion** — Upgraded `rss_ingest.py` to hit real supply chain feeds (Supply Chain Dive, Logistics Management) and parse headlines into structured JSON using LLM schema extraction.
- **Vector Store Roundtrip Tests** — Added comprehensive unit tests for `VectorStore` using `unittest.mock` to validate Pinecone namespace mapping and ingestion roundtripping without making external network calls.

### v0.2.0 — 2026-03-09 (Live Testing Fixes)

#### Bug Fixes
- **[Critical] JSON Truncation in Scenario Analysis** — The LLM was being asked to write the entire risk table as a single large JSON blob, causing response truncation even with high token limits. The `analyze_scenario` method in `scenario_engine.py` was refactored to a **two-step approach**: Python now computes the `risk_table` directly from Pinecone vector results, and the LLM is only asked to generate the smaller `briefing` and `ripple_effects` sections.
- **[Critical] Quick Toggle Buttons Not Working** — Streamlit's `text_area` widget manages its value via `st.session_state[widget_key]`. The toggle button callbacks were updating a separate `scenario_text` key that Streamlit ignores after first render. Fixed by writing directly to `st.session_state['scenario_input']`.
- **[Medium] Pinecone Package Rename** — Replaced deprecated `pinecone-client` with the official `pinecone` package (Pinecone's official SDK rename).
- **[Medium] LLM Token Limit** — Raised `max_output_tokens` from 2000 → 8192 across all providers (Anthropic, OpenAI, Gemini, Groq) to accommodate large scenario outputs.
- **[Low] Missing Plotly Dependency** — Added `plotly` to `requirements.txt`.
- **[Low] dotenv Override** — Added `override=True` to `load_dotenv()` so `.env` keys always take precedence when restarting.

#### Features
- **Middle East / Israel keyword mapping** — Scenarios mentioning "Israel", "Egypt", or "Middle East" now correctly route to the Hormuz disruption coefficient.

### v0.1.0 — 2026-03-08 (Initial Release)
- Full RAG architecture with Pinecone two-namespace design (`lead_times` + `disruptions`)
- Multi-provider LLM factory (Anthropic, OpenAI, Gemini, Groq, Ollama)
- Streamlit dashboard with Scenario Analyzer, Risk Dashboard, and Data Hub tabs
- The "Hormuz" Event Toggle (2026 geopolitical scenario)
- Graceful degradation to heuristic mode when Pinecone is offline
- Synthetic vendor lead time data generation (`data_gen.py`)

---

*Part of the **Mohith Kunta** supply chain AI portfolio — see also `dc_outbound_smoothing` (LevelSet) and `phantom_inventory` (Phantom Inventory Hunter).*

**Author:** Mohith Kunta  
**GitHub:** [https://github.com/m-kunta](https://github.com/m-kunta)
