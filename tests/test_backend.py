# Author: Mohith Kunta
# GitHub: https://github.com/m-kunta

import pytest
import pandas as pd
from scenario_engine import StrategicAnalystChain
import json

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
    
    assert chain._classify_risk(100, 100) == "Green" # 1.0
    assert chain._classify_risk(100, 110) == "Green" # 1.10
    assert chain._classify_risk(100, 120) == "Yellow" # 1.20
    assert chain._classify_risk(100, 134) == "Yellow" # 1.34
    assert chain._classify_risk(100, 136) == "Red" # 1.36
    assert chain._classify_risk(100, 200) == "Red" # 2.0

def test_disruption_coefficient_panama(mock_df):
    # Fallback should heuristically apply 1.35x if "Panama" is in query
    chain = StrategicAnalystChain(vector_store=None)
    
    # We patch the parsing to return a dummy to avoid LLM call for this test
    chain._parse_response = lambda x: {"briefing": {}, "ripple_effects": []}
    
    # We also mock get_llm_response globally if needed, but since we patched _parse_response, 
    # it still calls the actual LLM if we use 'Anthropic'. We can just inspect what it generates 
    # internally or mock the LLM
    import llm_providers
    original_get_llm_response = llm_providers.get_llm_response
    llm_providers.get_llm_response = lambda p, pr, m, s: "{}" # Mock
    
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
        # Restore mock
        llm_providers.get_llm_response = original_get_llm_response

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
    import llm_providers
    # Simulate LLM completely failing
    original_get_llm_response = llm_providers.get_llm_response
    def bad_llm(*args, **kwargs):
        raise ValueError("API Error")
    llm_providers.get_llm_response = bad_llm
    
    try:
        res = chain._fallback_analysis("panama", mock_df)
        assert res["source"] == "fallback"
        assert len(res["risk_table"]) == 1
        assert "executive_summary" in res["briefing"]
        assert "System operated in degraded fallback mode" in res["briefing"]["executive_summary"]
    finally:
        llm_providers.get_llm_response = original_get_llm_response
