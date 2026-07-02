# Project Gap Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the review gaps in PL Sourcing Co-Pilot so disruption scoring, RAG resilience, provider validation, dashboard coverage, and project docs/tests all match the product promises.

**Architecture:** Centralize disruption scenario classification into a small deterministic helper, then make both fallback and RAG paths consume that helper. Preserve the current Streamlit and module layout, but add normalization/validation at LLM and response boundaries so the UI always receives a safe schema.

**Tech Stack:** Python 3.9, Streamlit, pandas, Pinecone client, pytest, existing multi-provider LLM factory.

---

## File Structure

- Create: `disruption_model.py`
  - Owns disruption classification, coefficient selection, exposure flag mapping, and row metadata helpers.
- Modify: `scenario_engine.py`
  - Uses `disruption_model.py`; returns safe degraded RAG output when briefing generation fails; normalizes parsed LLM JSON.
- Modify: `vector_store.py`
  - Makes readiness less network-fragile; aligns disruption metadata shape with documented expectations where feasible.
- Modify: `llm_providers.py`
  - Adds provider/key validation and clearer error messages.
- Modify: `app.py`
  - Shows provider validation errors, includes Hormuz in dashboard exposure, and uses safer no-result messaging.
- Modify: `requirements.txt`
  - Adds test tooling or points to a dev requirements file.
- Create: `requirements-dev.txt`
  - Contains `pytest`.
- Modify: `README.md`, `HOW_TO_USE.md`, `AGENTS.md`
  - Aligns documented coefficients, mock disruption count, testing commands, and metadata behavior with code.
- Modify/Add tests under `tests/`
  - Adds coverage for each fixed gap.

---

### Task 1: Add Disruption Classification Unit

**Files:**
- Create: `disruption_model.py`
- Test: `tests/test_disruption_model.py`

- [ ] **Step 1: Write failing tests for classification, coefficients, and exposure flags**

Create `tests/test_disruption_model.py`:

```python
import pandas as pd

from disruption_model import classify_scenario, build_risk_row


def test_panama_50_percent_uses_severe_coefficient():
    disruption = classify_scenario("The Panama Canal has implemented a 50% transit reduction.")

    assert disruption.key == "panama"
    assert disruption.coefficient == 1.55
    assert disruption.exposure_flag == "panama_canal_exposure"


def test_panama_default_uses_standard_coefficient():
    disruption = classify_scenario("Panama Canal drought reduces vessel slots.")

    assert disruption.key == "panama"
    assert disruption.coefficient == 1.35


def test_supported_disruption_keywords():
    cases = {
        "red sea shipping suspended": ("suez", 1.40, "suez_canal_exposure"),
        "savannah port strike": ("savannah", 1.50, "savannah_port_exposure"),
        "west africa congestion at tema": ("west_africa", 1.25, "west_africa_port_exposure"),
        "strait of hormuz blocked": ("hrmz", 1.45, "hrmz_exposure"),
        "chittagong flooding disrupts cotton shipments": ("bangladesh_flooding", 1.30, "bangladesh_flooding_exposure"),
    }

    for text, expected in cases.items():
        disruption = classify_scenario(text)
        assert (disruption.key, disruption.coefficient, disruption.exposure_flag) == expected


def test_unknown_scenario_returns_none():
    assert classify_scenario("general supplier quality issue") is None


def test_build_risk_row_supports_derived_bangladesh_exposure():
    disruption = classify_scenario("Bangladesh flooding near Chittagong")
    row = pd.Series(
        {
            "vendor_name": "Test Vendor",
            "component": "Cotton",
            "category": "Apparel/Textiles",
            "origin_port": "Chittagong",
            "origin_country": "Bangladesh",
            "base_lead_days": 45,
        }
    )

    risk_row = build_risk_row(row, disruption)

    assert risk_row["vendor"] == "Test Vendor"
    assert risk_row["disruption_coefficient"] == 1.30
    assert risk_row["adjusted_lead_days"] == 58
    assert risk_row["risk_level"] == "Yellow"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_disruption_model.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'disruption_model'`.

- [ ] **Step 3: Implement `disruption_model.py`**

Create `disruption_model.py`:

```python
from dataclasses import dataclass
import re
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class Disruption:
    key: str
    coefficient: float
    exposure_flag: str
    rationale: str


def classify_risk(base_lead_days: int, adjusted_lead_days: int) -> str:
    ratio = adjusted_lead_days / base_lead_days if base_lead_days > 0 else 1.0
    if ratio > 1.35:
        return "Red"
    if ratio > 1.15:
        return "Yellow"
    return "Green"


def classify_scenario(scenario: str) -> Optional[Disruption]:
    text = (scenario or "").lower()
    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    percent = float(percent_match.group(1)) if percent_match else None

    if "panama" in text:
        coefficient = 1.55 if percent is not None and percent >= 50 else 1.35
        return Disruption(
            key="panama",
            coefficient=coefficient,
            exposure_flag="panama_canal_exposure",
            rationale="Exposed to Panama Canal transit constraints.",
        )
    if "suez" in text or "red sea" in text:
        return Disruption("suez", 1.40, "suez_canal_exposure", "Exposed to Suez Canal / Red Sea rerouting.")
    if "savannah" in text:
        return Disruption("savannah", 1.50, "savannah_port_exposure", "Exposed to Port of Savannah congestion/labor action.")
    if "tema" in text or "west africa" in text or "abidjan" in text:
        return Disruption("west_africa", 1.25, "west_africa_port_exposure", "Exposed to West Africa port congestion.")
    if "hormuz" in text or "strait" in text or "israel" in text or "egypt" in text or "middle east" in text:
        return Disruption("hrmz", 1.45, "hrmz_exposure", "Exposed to Strait of Hormuz disruption and fuel surcharges.")
    if "bangladesh" in text or "chittagong" in text:
        return Disruption("bangladesh_flooding", 1.30, "bangladesh_flooding_exposure", "Exposed to Bangladesh / Chittagong inland disruption.")
    return None


def has_exposure(row: Mapping[str, Any], disruption: Disruption) -> bool:
    if disruption.exposure_flag == "bangladesh_flooding_exposure":
        return (
            str(row.get("origin_country", "")).lower() == "bangladesh"
            or str(row.get("origin_port", "")).lower() == "chittagong"
        )
    return int(row.get(disruption.exposure_flag, 0) or 0) == 1


def build_risk_row(row: Mapping[str, Any], disruption: Disruption) -> Optional[dict]:
    if not has_exposure(row, disruption):
        return None

    base = int(float(row.get("base_lead_days", 0) or 0))
    adjusted = int(base * disruption.coefficient)
    return {
        "vendor": row.get("vendor_name", "Unknown"),
        "component": row.get("component", "Unknown"),
        "category": row.get("category", "Unknown"),
        "origin": f"{row.get('origin_port', '')}, {row.get('origin_country', '')}",
        "base_lead_days": base,
        "disruption_coefficient": disruption.coefficient,
        "adjusted_lead_days": adjusted,
        "risk_level": classify_risk(base, adjusted),
        "risk_rationale": disruption.rationale,
    }
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_disruption_model.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add disruption_model.py tests/test_disruption_model.py
git commit -m "feat: centralize disruption classification"
```

---

### Task 2: Refactor Scenario Engine to Use Shared Disruption Logic

**Files:**
- Modify: `scenario_engine.py`
- Modify: `tests/test_backend.py`
- Modify: `tests/test_scenario_engine_rag.py`

- [ ] **Step 1: Update tests for Panama severe coefficient and Bangladesh fallback**

Add to `tests/test_backend.py`:

```python
def test_panama_50_percent_uses_severe_coefficient(monkeypatch):
    _mock_llm(monkeypatch)
    chain = StrategicAnalystChain(vector_store=None)
    df = pd.DataFrame([{
        "vendor_name": "Panama Vendor",
        "component": "Furniture",
        "category": "Wood/Furniture",
        "origin_port": "Shenzhen",
        "origin_country": "China",
        "base_lead_days": 60,
        "panama_canal_exposure": 1,
        "suez_canal_exposure": 0,
        "savannah_port_exposure": 0,
        "west_africa_port_exposure": 0,
        "hrmz_exposure": 0,
    }])

    res = chain._fallback_analysis("Panama Canal 50% transit reduction", df)

    assert res["risk_table"][0]["disruption_coefficient"] == 1.55
    assert res["risk_table"][0]["adjusted_lead_days"] == 93
    assert res["risk_table"][0]["risk_level"] == "Red"


def test_bangladesh_flooding_supported_in_fallback(monkeypatch):
    _mock_llm(monkeypatch)
    chain = StrategicAnalystChain(vector_store=None)
    df = pd.DataFrame([{
        "vendor_name": "Bangladesh Vendor",
        "component": "Cotton",
        "category": "Apparel/Textiles",
        "origin_port": "Chittagong",
        "origin_country": "Bangladesh",
        "base_lead_days": 45,
    }])

    res = chain._fallback_analysis("Bangladesh flooding disrupts Chittagong roads", df)

    assert len(res["risk_table"]) == 1
    assert res["risk_table"][0]["disruption_coefficient"] == 1.30
```

- [ ] **Step 2: Run tests and verify new failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_backend.py::test_panama_50_percent_uses_severe_coefficient tests/test_backend.py::test_bangladesh_flooding_supported_in_fallback -q
```

Expected: fail because `scenario_engine.py` still uses old inline coefficient logic.

- [ ] **Step 3: Replace duplicated coefficient logic in `scenario_engine.py`**

Modify imports:

```python
from disruption_model import build_risk_row, classify_risk, classify_scenario
```

Replace `_classify_risk` body with:

```python
    def _classify_risk(self, base: int, adjusted: int) -> str:
        return classify_risk(base, adjusted)
```

In `_fallback_analysis`, replace the inline `scenario_lower` / `coeff_key` block and row coefficient logic with:

```python
        disruption = classify_scenario(scenario)

        risk_table = []
        if disruption:
            for _, row in raw_df.iterrows():
                risk_row = build_risk_row(row, disruption)
                if risk_row:
                    risk_table.append(risk_row)
```

In `analyze_scenario`, replace the inline `scenario_lower` / `coeff_key` block and item coefficient logic with:

```python
        disruption = classify_scenario(scenario)
        if not disruption:
            return self._fallback_analysis(scenario, raw_df)

        risk_table = []
        for item in lead_time_context:
            try:
                risk_row = build_risk_row(item, disruption)
                if risk_row:
                    risk_table.append(risk_row)
            except (ValueError, TypeError):
                continue
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_backend.py tests/test_scenario_engine_rag.py -q
```

Expected: pass except tests whose hardcoded expected Panama severe coefficient must be adjusted.

- [ ] **Step 5: Commit**

```bash
git add scenario_engine.py tests/test_backend.py tests/test_scenario_engine_rag.py
git commit -m "refactor: use shared disruption model in scenario engine"
```

---

### Task 3: Guarantee Safe Schema for LLM Responses and RAG Failures

**Files:**
- Modify: `scenario_engine.py`
- Test: `tests/test_scenario_engine_rag.py`
- Test: `tests/test_backend.py`

- [ ] **Step 1: Add failing tests for safe degraded RAG output and JSON normalization**

Add to `tests/test_scenario_engine_rag.py`:

```python
def test_analyze_scenario_rag_llm_failure_returns_degraded_schema(monkeypatch, raw_df):
    vs = FakeVectorStore(
        ready=True,
        lead_time_context=[{
            "vendor_name": "Vector Vendor",
            "component": "Wood Pulp",
            "category": "Wood/Furniture",
            "origin_port": "Genoa",
            "origin_country": "Italy",
            "base_lead_days": 45,
            "suez_canal_exposure": 1,
        }],
    )
    chain = StrategicAnalystChain(vector_store=vs)

    def bad_llm(*args, **kwargs):
        raise RuntimeError("provider offline")

    monkeypatch.setattr(scenario_engine.llm_providers, "get_llm_response", bad_llm)

    result = chain.analyze_scenario("red sea closure", raw_df)

    assert result["source"] == "rag+llm-degraded"
    assert result["risk_table"][0]["vendor"] == "Vector Vendor"
    assert "executive_summary" in result["briefing"]
    assert result["ripple_effects"]
```

Add to `tests/test_backend.py`:

```python
def test_parse_response_normalizes_missing_sections():
    chain = StrategicAnalystChain(vector_store=None)

    parsed = chain._parse_response('{"briefing": {"executive_summary": "Only summary"}}')

    assert parsed["risk_table"] == []
    assert parsed["ripple_effects"] == []
    assert parsed["briefing"]["executive_summary"] == "Only summary"
    assert parsed["briefing"]["key_findings"] == []
    assert parsed["briefing"]["recommended_actions"] == []
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_scenario_engine_rag.py::test_analyze_scenario_rag_llm_failure_returns_degraded_schema tests/test_backend.py::test_parse_response_normalizes_missing_sections -q
```

Expected: fail because RAG raises and `_parse_response` returns partial JSON.

- [ ] **Step 3: Implement response normalization and degraded RAG result**

In `scenario_engine.py`, add:

```python
    def _normalize_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        briefing = parsed.get("briefing") if isinstance(parsed.get("briefing"), dict) else {}
        parsed["risk_table"] = parsed.get("risk_table") if isinstance(parsed.get("risk_table"), list) else []
        parsed["ripple_effects"] = parsed.get("ripple_effects") if isinstance(parsed.get("ripple_effects"), list) else []
        parsed["briefing"] = {
            "executive_summary": briefing.get("executive_summary", ""),
            "key_findings": briefing.get("key_findings") if isinstance(briefing.get("key_findings"), list) else [],
            "affected_categories": briefing.get("affected_categories") if isinstance(briefing.get("affected_categories"), list) else [],
            "recommended_actions": briefing.get("recommended_actions") if isinstance(briefing.get("recommended_actions"), list) else [],
            "risk_horizon": briefing.get("risk_horizon", "Unknown"),
        }
        return parsed
```

Update `_parse_response` to return `self._normalize_response(parsed)` after successful `json.loads`.

Add:

```python
    def _degraded_result(self, scenario: str, risk_table: List[Dict[str, Any]], source: str, error: Exception) -> Dict[str, Any]:
        return {
            "risk_table": risk_table,
            "ripple_effects": [{
                "primary_disruption": scenario,
                "affected_route": "Multiple",
                "downstream_impacts": [f"Briefing generation failed: {error}"],
            }],
            "briefing": {
                "executive_summary": "Risk table generated successfully, but the LLM briefing could not be produced.",
                "key_findings": ["Structured risk scoring completed", "Narrative briefing unavailable"],
                "affected_categories": sorted({row.get("category", "Unknown") for row in risk_table}),
                "recommended_actions": ["Review the risk table and contact affected vendors directly"],
                "risk_horizon": "Unknown",
            },
            "source": source,
        }
```

Replace the RAG `except` block:

```python
        except Exception as e:
            return self._degraded_result(scenario, risk_table, "rag+llm-degraded", e)
```

Replace the fallback `except` block with `return self._degraded_result(scenario, risk_table, "fallback", e)`.

- [ ] **Step 4: Update old RAG failure test expectation**

In `tests/test_scenario_engine_rag.py`, remove or rewrite the old `with pytest.raises(...)` test so it asserts the degraded schema instead of an exception.

- [ ] **Step 5: Run scenario tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_backend.py tests/test_scenario_engine_rag.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add scenario_engine.py tests/test_backend.py tests/test_scenario_engine_rag.py
git commit -m "fix: return safe scenario schema on llm failures"
```

---

### Task 4: Improve RAG Coverage by Backfilling From Full Portfolio

**Files:**
- Modify: `scenario_engine.py`
- Test: `tests/test_scenario_engine_rag.py`

- [ ] **Step 1: Add failing test for vector top-k partial coverage**

Add to `tests/test_scenario_engine_rag.py`:

```python
def test_rag_backfills_exposed_rows_from_full_dataframe(monkeypatch):
    raw_df = pd.DataFrame([
        {
            "vendor_name": "Retrieved Vendor",
            "component": "Cotton",
            "category": "Apparel/Textiles",
            "origin_port": "Shenzhen",
            "origin_country": "China",
            "base_lead_days": 40,
            "panama_canal_exposure": 1,
        },
        {
            "vendor_name": "Missed Vendor",
            "component": "MDF",
            "category": "Wood/Furniture",
            "origin_port": "Mumbai",
            "origin_country": "India",
            "base_lead_days": 60,
            "panama_canal_exposure": 1,
        },
    ])
    vs = FakeVectorStore(
        ready=True,
        lead_time_context=[raw_df.iloc[0].to_dict()],
    )
    chain = StrategicAnalystChain(vector_store=vs)
    monkeypatch.setattr(
        scenario_engine.llm_providers,
        "get_llm_response",
        lambda *args, **kwargs: '{"ripple_effects": [], "briefing": {"executive_summary": "ok"}}',
    )

    result = chain.analyze_scenario("panama canal drought", raw_df)

    vendors = {row["vendor"] for row in result["risk_table"]}
    assert vendors == {"Retrieved Vendor", "Missed Vendor"}
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_scenario_engine_rag.py::test_rag_backfills_exposed_rows_from_full_dataframe -q
```

Expected: fail because only retrieved rows are scored.

- [ ] **Step 3: Add helper to score full dataframe and deduplicate rows**

In `scenario_engine.py`, add:

```python
    def _risk_table_from_rows(self, rows, disruption) -> List[Dict[str, Any]]:
        risk_table = []
        seen = set()
        for row in rows:
            try:
                risk_row = build_risk_row(row, disruption)
                if not risk_row:
                    continue
                key = (
                    risk_row["vendor"],
                    risk_row["component"],
                    risk_row["origin"],
                    risk_row["base_lead_days"],
                )
                if key in seen:
                    continue
                seen.add(key)
                risk_table.append(risk_row)
            except (ValueError, TypeError):
                continue
        return risk_table
```

Use it in RAG:

```python
        vector_rows = self._risk_table_from_rows(lead_time_context, disruption)
        full_rows = self._risk_table_from_rows((row for _, row in raw_df.iterrows()), disruption)
        risk_table = vector_rows + [
            row for row in full_rows
            if (row["vendor"], row["component"], row["origin"], row["base_lead_days"])
            not in {(r["vendor"], r["component"], r["origin"], r["base_lead_days"]) for r in vector_rows}
        ]
```

Sort and limit as before.

- [ ] **Step 4: Run RAG tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_scenario_engine_rag.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scenario_engine.py tests/test_scenario_engine_rag.py
git commit -m "fix: backfill rag risk scoring from full portfolio"
```

---

### Task 5: Add Provider Validation

**Files:**
- Modify: `llm_providers.py`
- Modify: `app.py`
- Test: `tests/test_support_modules.py`

- [ ] **Step 1: Add failing tests for provider validation**

Add to `tests/test_support_modules.py`:

```python
from llm_providers import validate_provider_config


def test_validate_provider_config_reports_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    error = validate_provider_config("OpenAI")

    assert error == "OpenAI selected but OPENAI_API_KEY is not set."


def test_validate_provider_config_accepts_ollama_without_key():
    assert validate_provider_config("Ollama") is None
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_support_modules.py::test_validate_provider_config_reports_missing_key tests/test_support_modules.py::test_validate_provider_config_accepts_ollama_without_key -q
```

Expected: fail because function does not exist.

- [ ] **Step 3: Implement validation**

In `llm_providers.py`, add:

```python
PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def validate_provider_config(provider: str) -> Optional[str]:
    provider_key = (provider or "").lower()
    env_key = PROVIDER_ENV_KEYS.get(provider_key)
    if provider_key == "ollama":
        return None
    if not env_key:
        return f"Unsupported LLM provider: {provider}"
    if not os.getenv(env_key):
        return f"{provider} selected but {env_key} is not set."
    return None
```

At the top of `get_llm_response`, add:

```python
    config_error = validate_provider_config(provider)
    if config_error:
        raise ValueError(config_error)
```

In `app.py`, import:

```python
from llm_providers import validate_provider_config
```

After the model input in the sidebar, add:

```python
    provider_error = validate_provider_config(st.session_state["provider"])
    if provider_error:
        st.error(provider_error)
```

Before calling `chain.analyze_scenario`, add:

```python
                    provider_error = validate_provider_config(st.session_state["provider"])
                    if provider_error:
                        st.error(provider_error)
                        st.stop()
```

- [ ] **Step 4: Run support tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_support_modules.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add llm_providers.py app.py tests/test_support_modules.py
git commit -m "fix: validate llm provider configuration"
```

---

### Task 6: Make Pinecone Readiness Less Fragile

**Files:**
- Modify: `vector_store.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: Add failing test for transient `list_indexes` failure after initialization**

Add to `tests/test_vector_store.py`:

```python
def test_is_ready_uses_cached_ready_state_after_index_initialized(monkeypatch):
    vs = VectorStore(api_key=None)
    vs._initialized = True
    vs._ready = True
    vs.index = FakeIndex()

    def broken_index_exists():
        raise RuntimeError("network hiccup")

    monkeypatch.setattr(vs, "_index_exists", broken_index_exists)

    assert vs.is_ready() is True
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_vector_store.py::test_is_ready_uses_cached_ready_state_after_index_initialized -q
```

Expected: fail because `_ready` is not used.

- [ ] **Step 3: Cache readiness**

In `VectorStore.__init__`, add:

```python
        self._ready = False
```

When existing index is found:

```python
                    self.index = self.pc.Index(self.index_name)
                    self._ready = True
```

In `init_index`, after assigning `self.index`, add:

```python
        self._ready = True
```

Replace `is_ready` with:

```python
    def is_ready(self) -> bool:
        """Returns True if Pinecone is configured and the index has been initialized."""
        if self._ready and self.index is not None:
            return True
        try:
            self._ready = self._initialized and self._index_exists()
        except Exception as e:
            print(f"Warning: Failed to check Pinecone readiness: {e}")
        return self._ready and self.index is not None
```

- [ ] **Step 4: Run vector tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_vector_store.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vector_store.py tests/test_vector_store.py
git commit -m "fix: cache pinecone readiness after initialization"
```

---

### Task 7: Align Disruption Metadata Shape

**Files:**
- Modify: `vector_store.py`
- Modify: `tests/test_vector_store.py`

- [ ] **Step 1: Decide metadata strategy**

Pinecone metadata may not reliably support arbitrary lists across all client/index versions. Keep the string field for compatibility and add a JSON string field for round-trip fidelity.

- [ ] **Step 2: Add failing test**

Update `test_ingest_disruptions_upserts_expected_namespace` in `tests/test_vector_store.py`:

```python
    assert upsert["vectors"][0]["metadata"]["affected_routes"] == "Asia-East Coast"
    assert upsert["vectors"][0]["metadata"]["affected_routes_json"] == '["Asia-East Coast"]'
```

- [ ] **Step 3: Run test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_vector_store.py::test_ingest_disruptions_upserts_expected_namespace -q
```

Expected: fail because `affected_routes_json` does not exist.

- [ ] **Step 4: Add JSON metadata field**

In `vector_store.py`, import `json`.

In `ingest_disruptions`, add:

```python
                "affected_routes_json": json.dumps(event["affected_routes"]),
```

- [ ] **Step 5: Run vector tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_vector_store.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add vector_store.py tests/test_vector_store.py
git commit -m "fix: preserve disruption route metadata as json"
```

---

### Task 8: Update Dashboard Coverage and UI States

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Include Hormuz exposure in dashboard chart**

In `app.py`, update `value_vars`:

```python
                value_vars=[
                    'panama_canal_exposure',
                    'suez_canal_exposure',
                    'savannah_port_exposure',
                    'west_africa_port_exposure',
                    'hrmz_exposure',
                ],
```

Then add a readable label mapping before plotting:

```python
            label_map = {
                "panama_canal": "Panama Canal",
                "suez_canal": "Suez Canal",
                "savannah_port": "Port Of Savannah",
                "west_africa_port": "West Africa Ports",
                "hrmz": "Strait Of Hormuz",
            }
            route_counts['Chokepoint'] = route_counts['Chokepoint'].str.replace('_exposure', '')
            route_counts['Chokepoint'] = route_counts['Chokepoint'].map(label_map).fillna(route_counts['Chokepoint'])
```

- [ ] **Step 2: Improve empty-data safety**

Before the analyze button logic, add:

```python
    if df.empty:
        st.warning("Lead time data is empty. Run `python data_gen.py` before analyzing scenarios.")
```

Disable the analyze button when `df.empty`:

```python
    if st.button("Analyze Scenario", type="primary", use_container_width=True, disabled=df.empty):
```

- [ ] **Step 3: Run Streamlit import smoke test**

Run:

```bash
.venv/bin/python -m py_compile app.py
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "fix: include hormuz exposure in dashboard"
```

---

### Task 9: Add Dev Requirements and Documentation Corrections

**Files:**
- Create: `requirements-dev.txt`
- Modify: `README.md`
- Modify: `HOW_TO_USE.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add dev requirements**

Create `requirements-dev.txt`:

```text
-r requirements.txt
pytest>=8.0.0
```

- [ ] **Step 2: Update test commands in docs**

Replace test command examples with:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

- [ ] **Step 3: Align coefficient docs**

Ensure docs list:

```text
Panama Canal -30% transit: 1.35
Panama Canal -50% transit: 1.55
Bangladesh flooding / Chittagong disruption: 1.30
Strait of Hormuz disruption: 1.45
```

- [ ] **Step 4: Align mock disruption count**

In `README.md`, change any “Mock Events (10)” claim to “Mock Events (6)”.

- [ ] **Step 5: Document route metadata compatibility**

Add a short note near disruption vector metadata:

```text
`affected_routes` is stored as a comma-separated string for Pinecone metadata compatibility; `affected_routes_json` preserves the original route list as JSON.
```

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt README.md HOW_TO_USE.md AGENTS.md
git commit -m "docs: align setup and disruption behavior"
```

---

### Task 10: Final Verification

**Files:**
- All changed files

- [ ] **Step 1: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run compile smoke test**

Run:

```bash
.venv/bin/python -m py_compile app.py scenario_engine.py vector_store.py llm_providers.py disruption_model.py rss_ingest.py data_gen.py
```

Expected: no output.

- [ ] **Step 3: Run fallback scenario smoke test**

Run:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
from scenario_engine import StrategicAnalystChain

df = pd.read_csv("data/vendor_lead_times.csv")
chain = StrategicAnalystChain(vector_store=None, provider="Ollama", model="llama3.2")
result = chain._fallback_analysis("Panama Canal 50% transit reduction", df)
print(result["source"])
print(result["risk_table"][0]["disruption_coefficient"] if result["risk_table"] else "no-risk")
PY
```

Expected: prints `fallback` and either `1.55` or `no-risk` depending on data exposure; it must not crash.

- [ ] **Step 4: Review git diff**

Run:

```bash
git diff --stat
git diff --check
```

Expected: no whitespace errors; diff only touches planned files.

- [ ] **Step 5: Final commit if needed**

If Task 10 produced fixes:

```bash
git add .
git commit -m "test: verify project gap fixes"
```

---

## Self-Review

- Spec coverage: Every review finding maps to a task: RAG resilience Task 3, Panama severe coefficient Task 1/2/9, duplicated keyword logic Task 1/2, RAG top-k portfolio miss Task 4, Pinecone readiness Task 6, provider validation Task 5, Hormuz dashboard Task 8, route metadata Task 7/9, dev test tooling Task 9.
- Placeholder scan: No task uses TBD-style placeholders; each code-changing task includes concrete test or implementation snippets.
- Type consistency: `Disruption`, `classify_scenario`, `classify_risk`, `has_exposure`, and `build_risk_row` are defined before all references.

