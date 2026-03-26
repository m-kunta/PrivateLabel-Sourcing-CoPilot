import sys
import types

import pandas as pd
import pytest

import vector_store
from vector_store import VectorStore


class FakeIndex:
    def __init__(self):
        self.upserts = []
        self.queries = []
        self.matches = []

    def upsert(self, vectors, namespace):
        self.upserts.append({"vectors": vectors, "namespace": namespace})

    def query(self, vector, namespace, top_k, include_metadata):
        self.queries.append(
            {
                "vector": vector,
                "namespace": namespace,
                "top_k": top_k,
                "include_metadata": include_metadata,
            }
        )
        return types.SimpleNamespace(matches=self.matches)


class FakePineconeClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.created_indexes = []
        self.index = FakeIndex()
        self.index_names = []

    def list_indexes(self):
        return [types.SimpleNamespace(name=name) for name in self.index_names]

    def create_index(self, **kwargs):
        self.created_indexes.append(kwargs)
        self.index_names.append(kwargs["name"])

    def Index(self, name):
        return self.index


def test_vector_store_init_without_key_is_not_ready():
    vs = VectorStore(api_key=None)
    assert vs.pc is None
    assert vs.index is None
    assert vs.is_ready() is False


def test_vector_store_init_with_existing_index(monkeypatch):
    fake_pinecone_module = types.SimpleNamespace(Pinecone=FakePineconeClient)
    monkeypatch.setitem(sys.modules, "pinecone", fake_pinecone_module)

    vs = VectorStore(api_key="test-key")
    vs.pc.index_names = [vs.index_name]
    vs.index = vs.pc.Index(vs.index_name)

    assert vs.pc.api_key == "test-key"
    assert vs._index_exists() is True


def test_get_embedder_is_lazy_and_embed_returns_vector(monkeypatch):
    class FakeEmbedding:
        def tolist(self):
            return [0.1, 0.2, 0.3]

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, text):
            assert text == "hello"
            return FakeEmbedding()

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    vs = VectorStore(api_key=None)
    assert vs.embed("hello") == [0.1, 0.2, 0.3]
    assert vs._embedder.model_name == "all-MiniLM-L6-v2"


def test_init_index_creates_pinecone_index(monkeypatch):
    fake_pinecone_module = types.SimpleNamespace(
        Pinecone=FakePineconeClient,
        ServerlessSpec=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "pinecone", fake_pinecone_module)

    vs = VectorStore(api_key="test-key")
    vs.pc.index_names = []
    vs.init_index()

    assert vs.pc.created_indexes[0]["name"] == vs.index_name
    assert vs.pc.created_indexes[0]["dimension"] == 384
    assert vs.index is vs.pc.index


def test_ingest_lead_times_builds_unique_vectors(monkeypatch, tmp_path):
    csv_path = tmp_path / "lead_times.csv"
    df = pd.DataFrame(
        [
            {
                "vendor_id": "V001",
                "vendor_name": "Vendor One",
                "brand_name": "BrandA",
                "category": "Packaging",
                "component": "Vanilla Extract",
                "origin_country": "Ghana",
                "origin_port": "Tema",
                "destination_port": "New York",
                "transport_mode": "Ocean Freight",
                "route_name": "West Africa-Atlantic",
                "base_lead_days": 40,
                "historical_variance_pct": 12.5,
                "panama_canal_exposure": 0,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 0,
                "west_africa_port_exposure": 1,
                "hrmz_exposure": 0,
            },
            {
                "vendor_id": "V001",
                "vendor_name": "Vendor One",
                "brand_name": "BrandA",
                "category": "Packaging",
                "component": "Vanilla Extract",
                "origin_country": "India",
                "origin_port": "Mumbai",
                "destination_port": "Savannah",
                "transport_mode": "Ocean Freight",
                "route_name": "Asia-East Coast",
                "base_lead_days": 55,
                "historical_variance_pct": 18.0,
                "panama_canal_exposure": 1,
                "suez_canal_exposure": 0,
                "savannah_port_exposure": 1,
                "west_africa_port_exposure": 0,
                "hrmz_exposure": 1,
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    vs = VectorStore(api_key=None)
    vs._initialized = True
    vs.pc = object()
    vs.index = FakeIndex()
    monkeypatch.setattr(vs, "_index_exists", lambda: True)
    monkeypatch.setattr(vs, "embed", lambda text: [len(text)])

    vs.ingest_lead_times(str(csv_path))

    assert len(vs.index.upserts) == 1
    batch = vs.index.upserts[0]
    assert batch["namespace"] == "lead_times"
    assert len(batch["vectors"]) == 2
    ids = [item["id"] for item in batch["vectors"]]
    assert len(set(ids)) == 2
    assert "West Africa ports" in batch["vectors"][0]["metadata"]["text"]
    assert "Panama Canal" in batch["vectors"][1]["metadata"]["text"]
    assert batch["vectors"][1]["metadata"]["base_lead_days"] == 55


def test_ingest_disruptions_upserts_expected_namespace(monkeypatch):
    vs = VectorStore(api_key=None)
    vs._initialized = True
    vs.pc = object()
    vs.index = FakeIndex()
    monkeypatch.setattr(vs, "_index_exists", lambda: True)
    monkeypatch.setattr(vs, "embed", lambda text: [0.5])

    vs.ingest_disruptions(
        [
            {
                "event_type": "Labor Action",
                "location": "Port of Savannah",
                "severity": "High",
                "affected_routes": ["Asia-East Coast"],
                "date": "2026-03-24",
                "source": "Mock Feed",
                "headline": "Savannah delays worsen",
                "text": "Operations are slowing sharply.",
            }
        ]
    )

    assert len(vs.index.upserts) == 1
    upsert = vs.index.upserts[0]
    assert upsert["namespace"] == "disruptions"
    assert upsert["vectors"][0]["metadata"]["affected_routes"] == "Asia-East Coast"


def test_query_returns_metadata(monkeypatch):
    vs = VectorStore(api_key=None)
    vs._initialized = True
    vs.pc = object()
    vs.index = FakeIndex()
    vs.index.matches = [types.SimpleNamespace(metadata={"vendor_name": "Vector Vendor"})]
    monkeypatch.setattr(vs, "_index_exists", lambda: True)
    monkeypatch.setattr(vs, "embed", lambda text: [1.0, 2.0])

    result = vs.query("panama", "lead_times", top_k=3)

    assert result == [{"vendor_name": "Vector Vendor"}]
    assert vs.index.queries[0]["namespace"] == "lead_times"
    assert vs.index.queries[0]["top_k"] == 3


def test_query_returns_empty_when_not_ready():
    vs = VectorStore(api_key=None)
    assert vs.query("anything", "lead_times") == []
