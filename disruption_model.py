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
        return Disruption(
            "suez",
            1.40,
            "suez_canal_exposure",
            "Exposed to Suez Canal / Red Sea rerouting.",
        )
    if "savannah" in text:
        return Disruption(
            "savannah",
            1.50,
            "savannah_port_exposure",
            "Exposed to Port of Savannah congestion/labor action.",
        )
    if "tema" in text or "west africa" in text or "abidjan" in text:
        return Disruption(
            "west_africa",
            1.25,
            "west_africa_port_exposure",
            "Exposed to West Africa port congestion.",
        )
    if "hormuz" in text or "strait" in text or "israel" in text or "egypt" in text or "middle east" in text:
        return Disruption(
            "hrmz",
            1.45,
            "hrmz_exposure",
            "Exposed to Strait of Hormuz disruption and fuel surcharges.",
        )
    if "bangladesh" in text or "chittagong" in text:
        return Disruption(
            "bangladesh_flooding",
            1.30,
            "bangladesh_flooding_exposure",
            "Exposed to Bangladesh / Chittagong inland disruption.",
        )
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
