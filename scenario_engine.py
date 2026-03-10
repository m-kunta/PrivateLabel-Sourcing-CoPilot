# Author: Mohith Kunta
# GitHub: https://github.com/m-kunta

import json
import re
import pandas as pd
from typing import Dict, Any, List, Optional
from vector_store import VectorStore
from llm_providers import get_llm_response

class StrategicAnalystChain:
    def __init__(self, vector_store: Optional[VectorStore], provider: str = "Anthropic", model: str = "claude-3-5-haiku-latest"):
        self.vs = vector_store
        self.provider = provider
        self.model = model
        
        # Base coefficients mapped to features from data_gen
        self.heuristic_coefficients = {
            "panama": 1.35, # Panama -30%
            "suez": 1.40,
            "savannah": 1.50,
            "west_africa": 1.25,
            "hrmz": 1.45  # The Hormuz Event Toggle
        }
        
        self.system_prompt = """You are a Principal Supply Chain Strategist specializing in private label raw material sourcing.
You have deep expertise in global freight routes, commodity supply chains, and disruption impact analysis.

When analyzing supply chain scenarios, you must:
1. Identify the primary disruption and which shipping routes / port chokepoints are affected.
2. Apply disruption coefficients to base lead times based on historical precedent.
3. Trace the "ripple effect" — second-order impacts on other components sharing the same route.
4. Classify each at-risk component by risk level: Red (>35% lead time increase), Yellow (15-35%), Green (<15%).
5. Generate a professional briefing suitable for a VP of Merchandising.

All outputs must strictly adhere to the following JSON schema:
{
    "risk_table": [
        {
            "vendor": str,
            "component": str,
            "category": str,
            "origin": str,
            "base_lead_days": int,
            "disruption_coefficient": float,
            "adjusted_lead_days": int,
            "risk_level": "Red" | "Yellow" | "Green",
            "risk_rationale": str
        }
    ],
    "ripple_effects": [
        {
            "primary_disruption": str,
            "affected_route": str,
            "downstream_impacts": [str]
        }
    ],
    "briefing": {
        "executive_summary": str,
        "key_findings": [str],
        "affected_categories": [str],
        "recommended_actions": [str],
        "risk_horizon": str
    },
    "source": "llm"
}

Always respond in valid JSON. Do not include markdown code fences in your response."""

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Safely parses JSON, stripping markdown code blocks if necessary."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
                
        raise ValueError(f"Failed to parse LLM response into JSON. Raw output: {text[:200]}...")

    def _classify_risk(self, base: int, adjusted: int) -> str:
        ratio = adjusted / base if base > 0 else 1.0
        if ratio > 1.35:
            return "Red"
        elif ratio > 1.15:
            return "Yellow"
        return "Green"

    def _fallback_analysis(self, scenario: str, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """Heuristic calculation when Pinecone is unavailable."""
        
        scenario_lower = scenario.lower()
        coeff_key = None
        if "panama" in scenario_lower: coeff_key = "panama"
        elif "suez" in scenario_lower or "red sea" in scenario_lower: coeff_key = "suez"
        elif "savannah" in scenario_lower: coeff_key = "savannah"
        elif "tema" in scenario_lower or "west africa" in scenario_lower or "abidjan" in scenario_lower: coeff_key = "west_africa"
        elif "hormuz" in scenario_lower: coeff_key = "hrmz"
        
        risk_table = []
        for _, row in raw_df.iterrows():
            coeff = 1.0
            rationale = "No direct route exposure detected."
            
            # Use .get() defensively for exposure flags that might be missing
            panama_exposure = row.get('panama_canal_exposure', 0)
            suez_exposure = row.get('suez_canal_exposure', 0)
            savannah_exposure = row.get('savannah_port_exposure', 0)
            wa_exposure = row.get('west_africa_port_exposure', 0)
            hrmz_exposure = row.get('hrmz_exposure', 0)
            
            if coeff_key == "panama" and panama_exposure == 1:
                coeff = self.heuristic_coefficients["panama"]
                rationale = "Exposed to Panama Canal drought/transit constraints."
            elif coeff_key == "suez" and suez_exposure == 1:
                coeff = self.heuristic_coefficients["suez"]
                rationale = "Exposed to Suez Canal / Red Sea rerouting."
            elif coeff_key == "savannah" and savannah_exposure == 1:
                coeff = self.heuristic_coefficients["savannah"]
                rationale = "Exposed to Port of Savannah congestion/labor action."
            elif coeff_key == "west_africa" and wa_exposure == 1:
                coeff = self.heuristic_coefficients["west_africa"]
                rationale = "Exposed to West Africa port congestion."
            elif coeff_key == "hrmz" and hrmz_exposure == 1:
                coeff = self.heuristic_coefficients["hrmz"]
                rationale = "Exposed to Strait of Hormuz blockage (+ fuel surcharge)."
                
            adj = int(row['base_lead_days'] * coeff)
            risk = self._classify_risk(int(row['base_lead_days']), adj)
            
            if coeff > 1.0:
                risk_table.append({
                    "vendor": row['vendor_name'],
                    "component": row['component'],
                    "category": row['category'],
                    "origin": f"{row['origin_port']}, {row['origin_country']}",
                    "base_lead_days": int(row['base_lead_days']),
                    "disruption_coefficient": coeff,
                    "adjusted_lead_days": adj,
                    "risk_level": risk,
                    "risk_rationale": rationale
                })
        
        risk_table = sorted(risk_table, key=lambda x: x["adjusted_lead_days"], reverse=True)[:15]

        # Use LLM to generate briefing based on heuristic table
        context = f"Heuristic Risk Table:\n{json.dumps(risk_table, indent=2)}\n\nScenario: {scenario}"
        prompt = f"Generate the ripple_effects and briefing based on this contextual risk table:\n\n{context}"
        
        try:
            response_text = get_llm_response(prompt, self.provider, self.model, self.system_prompt)
            parsed = self._parse_response(response_text)
            parsed["risk_table"] = risk_table
            parsed["source"] = "fallback"
            return parsed
        except Exception as e:
            return {
                "risk_table": risk_table,
                "ripple_effects": [{"primary_disruption": scenario, "affected_route": "Multiple", "downstream_impacts": [f"LLM Error: {e}"]}],
                "briefing": {
                    "executive_summary": "System operated in degraded fallback mode with LLM failure.",
                    "key_findings": ["LLM generation failed", "Heuristics applied directly"],
                    "affected_categories": [],
                    "recommended_actions": ["Review detailed risk table manually"],
                    "risk_horizon": "Unknown"
                },
                "source": "fallback"
            }

    def analyze_scenario(self, scenario: str, raw_df: pd.DataFrame) -> Dict[str, Any]:
        if not self.vs or not self.vs.is_ready():
            return self._fallback_analysis(scenario, raw_df)
            
        lead_time_context = self.vs.query(scenario, "lead_times", top_k=15)
        disruption_context = self.vs.query(scenario, "disruptions", top_k=5)
        
        # --- Step 1: Compute risk table in Python from retrieved vector data ---
        # Derive the disruption coefficient from the scenario text
        scenario_lower = scenario.lower()
        coeff_key = None
        if "panama" in scenario_lower: coeff_key = "panama"
        elif "suez" in scenario_lower or "red sea" in scenario_lower: coeff_key = "suez"
        elif "savannah" in scenario_lower: coeff_key = "savannah"
        elif "tema" in scenario_lower or "west africa" in scenario_lower or "abidjan" in scenario_lower: coeff_key = "west_africa"
        elif "hormuz" in scenario_lower or "strait" in scenario_lower: coeff_key = "hrmz"
        elif "israel" in scenario_lower or "egypt" in scenario_lower or "middle east" in scenario_lower: coeff_key = "hrmz"
        
        risk_table = []
        # Try to build risk table from retrieved vector context first
        for item in lead_time_context:
            try:
                base = int(float(item.get("base_lead_days", 30)))
                coeff = 1.0
                rationale = "No direct route exposure detected based on vector context."
                
                if coeff_key == "panama" and item.get("panama_canal_exposure", 0) == 1:
                    coeff = self.heuristic_coefficients["panama"]
                    rationale = "Exposed to Panama Canal constraints."
                elif coeff_key == "suez" and item.get("suez_canal_exposure", 0) == 1:
                    coeff = self.heuristic_coefficients["suez"]
                    rationale = "Exposed to Suez Canal / Red Sea rerouting."
                elif coeff_key == "savannah" and item.get("savannah_port_exposure", 0) == 1:
                    coeff = self.heuristic_coefficients["savannah"]
                    rationale = "Exposed to Port of Savannah."
                elif coeff_key == "west_africa" and item.get("west_africa_port_exposure", 0) == 1:
                    coeff = self.heuristic_coefficients["west_africa"]
                    rationale = "Exposed to West Africa ports."
                elif coeff_key == "hrmz" and item.get("hrmz_exposure", 0) == 1:
                    coeff = self.heuristic_coefficients["hrmz"]
                    rationale = "Exposed to Strait of Hormuz (+ fuel surcharge)."
                
                if coeff > 1.0:
                    adj = int(base * coeff)
                    risk_table.append({
                        "vendor": item.get("vendor_name", "Unknown"),
                        "component": item.get("component", "Unknown"),
                        "category": item.get("category", "Unknown"),
                        "origin": f"{item.get('origin_port', '')}, {item.get('origin_country', '')}",
                        "base_lead_days": base,
                        "disruption_coefficient": coeff,
                        "adjusted_lead_days": adj,
                        "risk_level": self._classify_risk(base, adj),
                        "risk_rationale": rationale
                    })
            except (ValueError, TypeError):
                continue
        
        # If no risk items from vector context, fall back to full df
        if not risk_table:
            return self._fallback_analysis(scenario, raw_df)
        
        risk_table = sorted(risk_table, key=lambda x: x["adjusted_lead_days"], reverse=True)[:15]
        
        # --- Step 2: Ask LLM only for the smaller briefing + ripple_effects ---
        briefing_system_prompt = """You are a Principal Supply Chain Strategist. 
Generate ONLY the ripple_effects and briefing sections based on the risk table provided.
Respond strictly with this JSON schema (no markdown fences):
{
    "ripple_effects": [{"primary_disruption": str, "affected_route": str, "downstream_impacts": [str]}],
    "briefing": {
        "executive_summary": str,
        "key_findings": [str],
        "affected_categories": [str],
        "recommended_actions": [str],
        "risk_horizon": str
    }
}"""
        context = f"Scenario: {scenario}\n\nRisk Table:\n{json.dumps(risk_table, indent=2)}\n\nDisruption Intelligence:\n{json.dumps([d.get('text','') for d in disruption_context], indent=2)}"
        prompt = f"Analyze this risk data and generate ripple_effects and briefing JSON:\n\n{context}"
        
        try:
            response_text = get_llm_response(prompt, self.provider, self.model, briefing_system_prompt)
            parsed = self._parse_response(response_text)
            parsed["risk_table"] = risk_table
            parsed["source"] = "rag+llm"
            return parsed
        except Exception as e:
            raise ValueError(f"Failed to generate scenario analysis: {e}")

