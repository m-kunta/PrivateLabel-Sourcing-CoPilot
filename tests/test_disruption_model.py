import pandas as pd

from disruption_model import build_risk_row, classify_scenario


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
