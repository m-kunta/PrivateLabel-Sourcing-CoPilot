# Author: Mohith Kunta
# GitHub: https://github.com/m-kunta

import pytest
import pandas as pd
from scenario_engine import StrategicAnalystChain
import scenario_engine


@pytest.fixture
def mock_df():
    return pd.DataFrame([
        {
            "vendor_name": "Test Vendor 1",
            "component": "Cotton",
            "category": "Textiles",
            "origin_port": "Shanghai",
            "origin_country": "China",
            "base_lead_days": 30,
            "panama_canal_exposure": 1,
            "suez_canal_exposure": 0,
            "savannah_port_exposure": 0,
            "west_africa_port_exposure": 0,
            "hrmz_exposure": 0
        },
        {
            "vendor_name": "Test Vendor 2",
            "component": "Cocoa",
            "category": "Food",
            "origin_port": "Abidjan",
            "origin_country": "Ivory Coast",
            "base_lead_days": 40,
            "panama_canal_exposure": 0,
            "suez_canal_exposure": 0,
            "savannah_port_exposure": 0,
            "west_africa_port_exposure": 1,
            "hrmz_exposure": 0
        }
    ])


def test_risk_classification_thresholds():
    chain = StrategicAnalystChain(vector_store=None)

    assert chain._classify_risk(100, 100) == "Green"   # 1.0
    assert chain._classify_risk(100, 110) == "Green"   # 1.10
    assert chain._classify_risk(100, 120) == "Yellow"  # 1.20
    assert chain._classify_risk(100, 134) == "Yellow"  # 1.34
    assert chain._classify_risk(100, 136) == "Red"     # 1.36
    assert chain._classify_risk(100, 200) == "Red"     # 2.0


def test_disruption_coefficient_panama(mock_df):
    # Fallback should heuristically apply 1.35x if "Panama" is in query
    chain = StrategicAnalystChain(vector_store=None)

    # We patch the parsing to return a dummy to avoid LLM call for this test
    chain._parse_response = lambda x: {"briefing": {}, "ripple_effects": []}

    original_get_llm_response = scenario_engine.llm_providers.get_llm_response
    scenario_engine.llm_providers.get_llm_response = lambda p, pr, m, s: "{}"  # Mock

    try:
        res = chain._fallback_analysis("panama canal is blocked", mock_df)

        # Test Vendor 1 has panama exposure, base is 30, 30 * 1.35 = 40
        rt = res["risk_table"]
        assert len(rt) == 1
        assert rt[0]["vendor"] == "Test Vendor 1"
        assert rt[0]["disruption_coefficient"] == 1.35
        assert rt[0]["adjusted_lead_days"] == 40
        assert rt[0]["risk_level"] == "Yellow"
    finally:
        scenario_engine.llm_providers.get_llm_response = original_get_llm_response


def test_json_schema_compliance():
    chain = StrategicAnalystChain(vector_store=None)

    raw_markdown = '''
Here is your analysis:
```json
{
  "risk_table": [],
  "ripple_effects": [],
  "briefing": {
    "executive_summary": "Test"
  }
}
```
Have a good day!
    '''

    parsed = chain._parse_response(raw_markdown)
    assert parsed["briefing"]["executive_summary"] == "Test"
    assert type(parsed["risk_table"]) is list


def test_fallback_schema_fallback(mock_df):
    chain = StrategicAnalystChain(vector_store=None)
    original_get_llm_response = scenario_engine.llm_providers.get_llm_response

    def bad_llm(*args, **kwargs):
        raise ValueError("API Error")

    scenario_engine.llm_providers.get_llm_response = bad_llm

    try:
        res = chain._fallback_analysis("panama", mock_df)
        assert res["source"] == "fallback"
        assert len(res["risk_table"]) == 1
        assert "executive_summary" in res["briefing"]
        assert "System operated in degraded fallback mode" in res["briefing"]["executive_summary"]
    finally:
        scenario_engine.llm_providers.get_llm_response = original_get_llm_response


# ---------------------------------------------------------------------------
# Helpers shared by the new test classes
# ---------------------------------------------------------------------------

def _mock_llm(monkeypatch):
    """Patch llm_providers.get_llm_response with a no-op returning minimal JSON."""
    monkeypatch.setattr(scenario_engine.llm_providers, "get_llm_response", lambda *a, **kw: "{}")


# ---------------------------------------------------------------------------
# TestFallbackDisruptionCoefficients
# Verifies that each of the 5 heuristic disruption types applies the correct
# coefficient and that only rows with the matching exposure flag are included.
# ---------------------------------------------------------------------------

class TestFallbackDisruptionCoefficients:

    def test_suez_coefficient(self, monkeypatch):
        """'red sea closure' → suez-exposed row → coefficient 1.40, risk Red."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)
        df = pd.DataFrame([{
            "vendor_name": "Suez Vendor",
            "component": "Olive Oil",
            "category": "Food",
            "origin_port": "Alexandria",
            "origin_country": "Egypt",
            "base_lead_days": 50,
            "panama_canal_exposure": 0,
            "suez_canal_exposure": 1,
            "savannah_port_exposure": 0,
            "west_africa_port_exposure": 0,
            "hrmz_exposure": 0,
        }])
        res = chain._fallback_analysis("red sea closure", df)
        rt = res["risk_table"]
        assert len(rt) == 1, "Suez-exposed row should appear in risk_table"
        assert rt[0]["disruption_coefficient"] == 1.40
        assert rt[0]["adjusted_lead_days"] == int(50 * 1.40)
        assert rt[0]["risk_level"] == "Red"   # 1.40 > 1.35

    def test_savannah_coefficient(self, monkeypatch):
        """'savannah port strike' → savannah-exposed row → coefficient 1.50, risk Red."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)
        df = pd.DataFrame([{
            "vendor_name": "Savannah Vendor",
            "component": "Lumber",
            "category": "Building Materials",
            "origin_port": "Savannah",
            "origin_country": "USA",
            "base_lead_days": 20,
            "panama_canal_exposure": 0,
            "suez_canal_exposure": 0,
            "savannah_port_exposure": 1,
            "west_africa_port_exposure": 0,
            "hrmz_exposure": 0,
        }])
        res = chain._fallback_analysis("savannah port strike", df)
        rt = res["risk_table"]
        assert len(rt) == 1
        assert rt[0]["disruption_coefficient"] == 1.50
        assert rt[0]["adjusted_lead_days"] == int(20 * 1.50)
        assert rt[0]["risk_level"] == "Red"   # 1.50 > 1.35

    def test_west_africa_coefficient(self, monkeypatch):
        """'west africa port congestion' → wa-exposed row → coefficient 1.25, risk Yellow."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)
        df = pd.DataFrame([{
            "vendor_name": "WA Vendor",
            "component": "Cocoa",
            "category": "Food",
            "origin_port": "Tema",
            "origin_country": "Ghana",
            "base_lead_days": 40,
            "panama_canal_exposure": 0,
            "suez_canal_exposure": 0,
            "savannah_port_exposure": 0,
            "west_africa_port_exposure": 1,
            "hrmz_exposure": 0,
        }])
        res = chain._fallback_analysis("west africa port congestion", df)
        rt = res["risk_table"]
        assert len(rt) == 1
        assert rt[0]["disruption_coefficient"] == 1.25
        assert rt[0]["adjusted_lead_days"] == int(40 * 1.25)
        assert rt[0]["risk_level"] == "Yellow"  # 1.25 > 1.15, not > 1.35

    def test_hormuz_coefficient(self, monkeypatch):
        """'hormuz blockage' → hrmz-exposed row → coefficient 1.45, risk Red."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)
        df = pd.DataFrame([{
            "vendor_name": "Gulf Vendor",
            "component": "Petroleum Wax",
            "category": "Chemicals",
            "origin_port": "Dubai",
            "origin_country": "UAE",
            "base_lead_days": 35,
            "panama_canal_exposure": 0,
            "suez_canal_exposure": 0,
            "savannah_port_exposure": 0,
            "west_africa_port_exposure": 0,
            "hrmz_exposure": 1,
        }])
        res = chain._fallback_analysis("hormuz blockage", df)
        rt = res["risk_table"]
        assert len(rt) == 1
        assert rt[0]["disruption_coefficient"] == 1.45
        assert rt[0]["adjusted_lead_days"] == int(35 * 1.45)
        assert rt[0]["risk_level"] == "Red"   # 1.45 > 1.35

    def test_no_exposure_row_excluded(self, monkeypatch):
        """A row with panama_canal_exposure=0 must NOT appear for a Panama query."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)
        df = pd.DataFrame([{
            "vendor_name": "Safe Vendor",
            "component": "Steel",
            "category": "Industrial",
            "origin_port": "Rotterdam",
            "origin_country": "Netherlands",
            "base_lead_days": 25,
            "panama_canal_exposure": 0,   # no Panama exposure
            "suez_canal_exposure": 0,
            "savannah_port_exposure": 0,
            "west_africa_port_exposure": 0,
            "hrmz_exposure": 0,
        }])
        res = chain._fallback_analysis("panama canal drought", df)
        assert res["risk_table"] == [], "Unexposed row should not appear in risk_table"

    def test_unrecognized_scenario_empty_table(self, monkeypatch):
        """A scenario with no recognised keyword produces an empty risk_table."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)
        # All exposure flags set, so if any keyword erroneously matches, the table won't be empty
        df = pd.DataFrame([{
            "vendor_name": "Any Vendor",
            "component": "Widget",
            "category": "General",
            "origin_port": "Hamburg",
            "origin_country": "Germany",
            "base_lead_days": 20,
            "panama_canal_exposure": 1,
            "suez_canal_exposure": 1,
            "savannah_port_exposure": 1,
            "west_africa_port_exposure": 1,
            "hrmz_exposure": 1,
        }])
        res = chain._fallback_analysis("random weather event in an unknown location", df)
        assert res["risk_table"] == [], "Unrecognised scenario should produce no risk rows"


# ---------------------------------------------------------------------------
# TestRippleEffectSavannah  (P1 — CLAUDE.md)
# "test_ripple_effect_savannah — Savannah strike impacts ≥ 2 component categories"
# ---------------------------------------------------------------------------

class TestRippleEffectSavannah:

    def test_savannah_impacts_multiple_categories(self, monkeypatch):
        """
        A Savannah port strike surfaces risk rows across ≥ 2 distinct product
        categories, demonstrating the cross-lane ripple effect.
        """
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)

        df = pd.DataFrame([
            {
                "vendor_name": "Apparel Co",
                "component": "Cotton Yarn",
                "category": "Apparel/Textiles",
                "origin_port": "Chittagong",
                "origin_country": "Bangladesh",
                "base_lead_days": 45,
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 1,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 0,
            },
            {
                "vendor_name": "Food Ingredients Ltd",
                "component": "Sesame Seeds",
                "category": "Food/Beverage",
                "origin_port": "Lagos",
                "origin_country": "Nigeria",
                "base_lead_days": 30,
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 1,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 0,
            },
            {
                "vendor_name": "Beauty Supply Inc",
                "component": "Shea Butter",
                "category": "Health & Beauty",
                "origin_port": "Accra",
                "origin_country": "Ghana",
                "base_lead_days": 35,
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 1,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 0,
            },
        ])

        res = chain._fallback_analysis("savannah port labor strike", df)
        rt = res["risk_table"]

        assert len(rt) >= 2, "At least 2 Savannah-exposed rows should appear in risk_table"

        categories_hit = {r["category"] for r in rt}
        assert len(categories_hit) >= 2, (
            f"Savannah strike should impact ≥ 2 categories; got: {categories_hit}"
        )

        for row in rt:
            assert row["disruption_coefficient"] == 1.50
            assert row["risk_level"] == "Red"   # 1.50 > 1.35


# ---------------------------------------------------------------------------
# TestRiskTableOrdering
# Verifies that _fallback_analysis returns risk_table sorted by
# adjusted_lead_days descending (as per the sorted(...)[:15] in engine).
# ---------------------------------------------------------------------------

class TestRiskTableOrdering:

    def test_risk_table_sorted_descending(self, monkeypatch):
        """risk_table must be ordered by adjusted_lead_days descending."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)

        df = pd.DataFrame([
            {
                "vendor_name": "Short Lead Vendor",
                "component": "Labels",
                "category": "Packaging",
                "origin_port": "Guangzhou",
                "origin_country": "China",
                "base_lead_days": 10,       # adj = 15
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 1,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 0,
            },
            {
                "vendor_name": "Long Lead Vendor",
                "component": "Fabric",
                "category": "Textiles",
                "origin_port": "Mumbai",
                "origin_country": "India",
                "base_lead_days": 60,       # adj = 90
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 1,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 0,
            },
            {
                "vendor_name": "Medium Lead Vendor",
                "component": "Buttons",
                "category": "Accessories",
                "origin_port": "Ho Chi Minh",
                "origin_country": "Vietnam",
                "base_lead_days": 30,       # adj = 45
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 1,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 0,
            },
        ])

        res = chain._fallback_analysis("savannah port strike", df)
        rt = res["risk_table"]

        assert len(rt) == 3
        adj_days = [r["adjusted_lead_days"] for r in rt]
        assert adj_days == sorted(adj_days, reverse=True), (
            f"risk_table should be sorted descending by adjusted_lead_days; got {adj_days}"
        )


# ---------------------------------------------------------------------------
# TestRiskThresholdBoundaryPin
# Pins the exact boundary between Yellow and Red at ratio = 1.35, and between
# Green and Yellow at 1.15. Both thresholds use strict '>' so boundary values
# fall into the lower bucket.
# ---------------------------------------------------------------------------

class TestRiskThresholdBoundaryPin:

    def test_exactly_135_is_yellow_not_red(self):
        """ratio == 1.35 exactly → Yellow (condition is strictly > 1.35)."""
        chain = StrategicAnalystChain(vector_store=None)
        assert chain._classify_risk(100, 135) == "Yellow"

    def test_one_above_135_is_red(self):
        """ratio 1.36 → Red."""
        chain = StrategicAnalystChain(vector_store=None)
        assert chain._classify_risk(100, 136) == "Red"

    def test_exactly_115_is_green_not_yellow(self):
        """ratio == 1.15 exactly → Green (condition is strictly > 1.15)."""
        chain = StrategicAnalystChain(vector_store=None)
        assert chain._classify_risk(100, 115) == "Green"

    def test_one_above_115_is_yellow(self):
        """ratio 1.16 → Yellow."""
        chain = StrategicAnalystChain(vector_store=None)
        assert chain._classify_risk(100, 116) == "Yellow"


# ---------------------------------------------------------------------------
# TestMiddleEastKeywordRouting
# Israel, Egypt, and 'middle east' keywords now route to hrmz in BOTH
# _fallback_analysis and analyze_scenario (RAG path). Behavior is consistent.
# ---------------------------------------------------------------------------

class TestMiddleEastKeywordRouting:

    def test_israel_keyword_routes_to_hrmz_in_fallback(self, monkeypatch):
        """'israel' now routes to hrmz coefficient 1.45 in _fallback_analysis."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)
        df = pd.DataFrame([{
            "vendor_name": "Gulf Vendor",
            "component": "Petroleum Wax",
            "category": "Chemicals",
            "origin_port": "Dubai",
            "origin_country": "UAE",
            "base_lead_days": 35,
            "panama_canal_exposure": 0,
            "suez_canal_exposure": 0,
            "savannah_port_exposure": 0,
            "west_africa_port_exposure": 0,
            "hrmz_exposure": 1,
        }])
        res = chain._fallback_analysis("israel conflict escalation", df)
        assert len(res["risk_table"]) == 1
        assert res["risk_table"][0]["disruption_coefficient"] == 1.45
        assert res["risk_table"][0]["risk_level"] == "Red"

    def test_egypt_keyword_routes_to_hrmz_in_fallback(self, monkeypatch):
        """'egypt' now routes to hrmz coefficient 1.45 in _fallback_analysis."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)
        df = pd.DataFrame([{
            "vendor_name": "Suez Vendor",
            "component": "Cotton",
            "category": "Textiles",
            "origin_port": "Alexandria",
            "origin_country": "Egypt",
            "base_lead_days": 45,
            "panama_canal_exposure": 0,
            "suez_canal_exposure": 0,
            "savannah_port_exposure": 0,
            "west_africa_port_exposure": 0,
            "hrmz_exposure": 1,
        }])
        res = chain._fallback_analysis("egypt port tensions", df)
        assert len(res["risk_table"]) == 1
        assert res["risk_table"][0]["disruption_coefficient"] == 1.45
        assert res["risk_table"][0]["risk_level"] == "Red"

    def test_middle_east_keyword_routes_to_hrmz_in_fallback(self, monkeypatch):
        """'middle east' now routes to hrmz coefficient 1.45 in _fallback_analysis."""
        _mock_llm(monkeypatch)
        chain = StrategicAnalystChain(vector_store=None)
        df = pd.DataFrame([{
            "vendor_name": "Gulf Vendor",
            "component": "Shea Butter",
            "category": "Health & Beauty",
            "origin_port": "Muscat",
            "origin_country": "Oman",
            "base_lead_days": 28,
            "panama_canal_exposure": 0,
            "suez_canal_exposure": 0,
            "savannah_port_exposure": 0,
            "west_africa_port_exposure": 0,
            "hrmz_exposure": 1,
        }])
        res = chain._fallback_analysis("middle east tensions escalate", df)
        assert len(res["risk_table"]) == 1
        assert res["risk_table"][0]["disruption_coefficient"] == 1.45
        assert res["risk_table"][0]["risk_level"] == "Red"

# ---------------------------------------------------------------------------
# TestVectorStoreRoundtrip
# Verifies the P1 requirement: ingest -> query returns matching record.
# Mocks the external Pinecone API to ensure unit test speed and reliability.
# ---------------------------------------------------------------------------

class TestVectorStoreRoundtrip:

    def test_vector_store_ingest_roundtrip(self, monkeypatch):
        from unittest.mock import MagicMock
        from vector_store import VectorStore
        
        # 1. Setup mock Pinecone Index
        mock_index = MagicMock()
        
        # 2. Initialize VectorStore with a dummy key
        vs = VectorStore(api_key="dummy")
        vs.index_name = "test-index"
        vs.pc = MagicMock()
        vs.index = mock_index
        vs._initialized = True
        
        # Fake index exists
        mock_idx_obj = MagicMock()
        mock_idx_obj.name = "test-index"
        vs.pc.list_indexes.return_value = [mock_idx_obj]
        
        assert vs.is_ready() is True, "Mock VectorStore should be ready"
        
        # 3. Ingest mock disruption
        disruption = [{
            "event_type": "Test Event",
            "location": "Test Port",
            "severity": "High",
            "affected_routes": ["Route A"],
            "date": "2026-04-26",
            "source": "Test Source",
            "headline": "Test Headline",
            "text": "Test body text"
        }]
        
        # Mock embed to avoid downloading sentence-transformers
        monkeypatch.setattr(vs, "embed", lambda text: [0.1, 0.2, 0.3])
        
        vs.ingest_disruptions(disruption)
        
        # Verify upsert was called correctly
        mock_index.upsert.assert_called_once()
        upsert_kwargs = mock_index.upsert.call_args.kwargs
        assert upsert_kwargs["namespace"] == "disruptions"
        vectors = upsert_kwargs["vectors"]
        assert len(vectors) == 1
        assert vectors[0]["metadata"]["headline"] == "Test Headline"
        
        # 4. Mock query response
        mock_query_response = MagicMock()
        mock_match = MagicMock()
        mock_match.metadata = vectors[0]["metadata"]
        mock_query_response.matches = [mock_match]
        mock_index.query.return_value = mock_query_response
        
        # 5. Query and verify it returns the ingested record
        results = vs.query("test query", namespace="disruptions")
        
        mock_index.query.assert_called_once()
        query_kwargs = mock_index.query.call_args.kwargs
        assert query_kwargs["namespace"] == "disruptions"
        assert len(results) == 1
        assert results[0]["headline"] == "Test Headline"
        assert results[0]["event_type"] == "Test Event"
