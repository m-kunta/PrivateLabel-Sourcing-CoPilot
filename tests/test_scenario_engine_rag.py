import pandas as pd
import pytest

import scenario_engine
from scenario_engine import StrategicAnalystChain


class FakeVectorStore:
    def __init__(self, ready=True, lead_time_context=None, disruption_context=None):
        self._ready = ready
        self.lead_time_context = lead_time_context or []
        self.disruption_context = disruption_context or []
        self.calls = []

    def is_ready(self):
        return self._ready

    def query(self, query_text, namespace, top_k=5):
        self.calls.append((query_text, namespace, top_k))
        if namespace == "lead_times":
            return self.lead_time_context
        if namespace == "disruptions":
            return self.disruption_context
        return []


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        [
            {
                "vendor_name": "Fallback Vendor",
                "component": "Cotton",
                "category": "Textiles",
                "origin_port": "Shanghai",
                "origin_country": "China",
                "base_lead_days": 30,
                "panama_canal_exposure": 1,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 0,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 0,
            }
        ]
    )


def test_analyze_scenario_rag_success(monkeypatch, raw_df):
    vs = FakeVectorStore(
        ready=True,
        lead_time_context=[
            {
                "vendor_name": "Vector Vendor",
                "component": "Plastic Resins",
                "category": "Packaging",
                "origin_port": "Salalah",
                "origin_country": "Oman",
                "base_lead_days": 50,
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 0,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 1,
            }
        ],
        disruption_context=[{"text": "Hormuz traffic disruptions intensify."}],
    )
    chain = StrategicAnalystChain(vector_store=vs)

    monkeypatch.setattr(
        scenario_engine.llm_providers,
        "get_llm_response",
        lambda *args, **kwargs: """
        {
          "ripple_effects": [
            {
              "primary_disruption": "Hormuz blockage",
              "affected_route": "Middle East-Atlantic",
              "downstream_impacts": ["Fuel surcharges rise"]
            }
          ],
          "briefing": {
            "executive_summary": "Risk is elevated.",
            "key_findings": ["One component exposed"],
            "affected_categories": ["Packaging"],
            "recommended_actions": ["Book alternate capacity"],
            "risk_horizon": "2-4 weeks"
          }
        }
        """,
    )

    result = chain.analyze_scenario("middle east shipping disruption", raw_df)

    assert result["source"] == "rag+llm"
    assert result["briefing"]["executive_summary"] == "Risk is elevated."
    assert result["risk_table"][0]["vendor"] == "Vector Vendor"
    assert result["risk_table"][0]["disruption_coefficient"] == 1.45
    assert result["risk_table"][0]["risk_level"] == "Red"
    assert vs.calls == [
        ("middle east shipping disruption", "lead_times", 15),
        ("middle east shipping disruption", "disruptions", 5),
    ]


def test_analyze_scenario_rag_no_exposed_matches_falls_back(monkeypatch, raw_df):
    vs = FakeVectorStore(
        ready=True,
        lead_time_context=[
            {
                "vendor_name": "Vector Vendor",
                "component": "Unrelated",
                "category": "General",
                "origin_port": "Hamburg",
                "origin_country": "Germany",
                "base_lead_days": 20,
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 0,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 0,
            }
        ],
    )
    chain = StrategicAnalystChain(vector_store=vs)

    def fake_fallback(scenario, df):
        return {"source": "fallback", "risk_table": ["used-fallback"]}

    monkeypatch.setattr(chain, "_fallback_analysis", fake_fallback)

    result = chain.analyze_scenario("panama canal drought", raw_df)

    assert result == {"source": "fallback", "risk_table": ["used-fallback"]}


def test_analyze_scenario_rag_llm_failure_is_wrapped(monkeypatch, raw_df):
    vs = FakeVectorStore(
        ready=True,
        lead_time_context=[
            {
                "vendor_name": "Vector Vendor",
                "component": "Wood Pulp",
                "category": "Wood/Furniture",
                "origin_port": "Genoa",
                "origin_country": "Italy",
                "base_lead_days": 45,
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 1,
                "savannah_port_exposure": 0,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 0,
            }
        ],
    )
    chain = StrategicAnalystChain(vector_store=vs)

    def bad_llm(*args, **kwargs):
        raise RuntimeError("provider offline")

    monkeypatch.setattr(scenario_engine.llm_providers, "get_llm_response", bad_llm)

    with pytest.raises(ValueError, match="Failed to generate scenario analysis: provider offline"):
        chain.analyze_scenario("red sea closure", raw_df)


def test_parse_response_invalid_json_raises():
    chain = StrategicAnalystChain(vector_store=None)

    with pytest.raises(ValueError, match="Failed to parse LLM response into JSON"):
        chain._parse_response("not json at all")
