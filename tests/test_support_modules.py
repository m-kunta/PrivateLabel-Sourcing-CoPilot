import os

import pandas as pd
import pytest

from data_gen import generate_synthetic_data
from llm_providers import get_llm_response
from rss_ingest import get_mock_disruptions


def test_get_mock_disruptions_shape():
    disruptions = get_mock_disruptions()

    assert len(disruptions) == 6
    assert all("headline" in item for item in disruptions)
    assert all("affected_routes" in item for item in disruptions)


def test_generate_synthetic_data_writes_expected_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    generate_synthetic_data(num_records=8, seed=7)

    output_path = tmp_path / "data" / "vendor_lead_times.csv"
    assert output_path.exists()

    df = pd.read_csv(output_path)
    assert len(df) == 8
    assert "vendor_id" in df.columns
    assert "hrmz_exposure" in df.columns


def test_get_llm_response_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported LLM provider: UnknownProvider"):
        get_llm_response("prompt", "UnknownProvider", "model")
