# Author: Mohith Kunta
# GitHub: https://github.com/m-kunta

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "pl-sourcing-copilot")
        self.pc = None
        self.index = None
        self._embedder = None
        self._initialized = False
        
        if self.api_key:
            try:
                from pinecone import Pinecone
                self.pc = Pinecone(api_key=self.api_key)
                if self._index_exists():
                    self.index = self.pc.Index(self.index_name)
                self._initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize Pinecone: {e}")
                
    def _get_embedder(self):
        if self._embedder is None:
            # Lazy import to speed up initial app load if Pinecone isn't used
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def embed(self, text: str) -> List[float]:
        return self._get_embedder().encode(text).tolist()

    def _index_exists(self) -> bool:
        if not self.pc:
            return False
        return self.index_name in [idx.name for idx in self.pc.list_indexes()]

    def is_ready(self) -> bool:
        """Returns True if Pinecone is configured and the index exists."""
        return self._initialized and self._index_exists()

    def init_index(self):
        """Creates the Pinecone index if it doesn't exist."""
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is not set.")
        
        from pinecone import ServerlessSpec
        
        if not self._index_exists():
            print(f"Creating index '{self.index_name}'... This may take a minute.")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.pc.Index(self.index_name)
        print(f"✅ Index '{self.index_name}' is ready.")

    def ingest_lead_times(self, csv_path: str):
        """Embeds and uploads lead time data to the 'lead_times' namespace."""
        if not self.is_ready():
            raise ValueError("VectorStore is not ready. Call init_index() first.")
            
        df = pd.read_csv(csv_path)
        vectors = []
        
        for idx, row in df.iterrows():
            # Create a rich text representation for semantic search
            text = (
                f"Vendor: {row['vendor_name']} ({row['brand_name']}). "
                f"Category: {row['category']}. Component: {row['component']}. "
                f"Route: {row['route_name']} from {row['origin_port']}, {row['origin_country']} "
                f"to {row['destination_port']} via {row['transport_mode']}. "
                f"Base lead time: {row['base_lead_days']} days with {row['historical_variance_pct']}% variance."
            )
            
            # Additional exposure context
            if row['panama_canal_exposure'] == 1: text += " Exposed to Panama Canal."
            if row['suez_canal_exposure'] == 1: text += " Exposed to Suez Canal."
            if row['savannah_port_exposure'] == 1: text += " Exposed to Port of Savannah."
            if row['west_africa_port_exposure'] == 1: text += " Exposed to West Africa ports."
            if row.get('hrmz_exposure', 0) == 1: text += " Exposed to Strait of Hormuz."

            # Clean up numpy types for JSON serialization
            metadata = {}
            for k, v in row.items():
                if pd.isna(v):
                    metadata[k] = None
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    metadata[k] = float(v) if isinstance(v, (float, np.floating)) else int(v)
                else:
                    metadata[k] = str(v)
            metadata["text"] = text
            
            # Unique ID
            vid = str(row['vendor_id']).replace(" ", "")
            comp = str(row['component']).replace(" ", "_").replace("/", "_")
            route = str(row['route_name']).replace(" ", "_").replace("/", "_")
            origin = str(row['origin_port']).replace(" ", "_").replace("/", "_")
            destination = str(row['destination_port']).replace(" ", "_").replace("/", "_")
            v_id = f"lt_{vid}_{comp}_{route}_{origin}_{destination}_{idx}"
            
            vectors.append({
                "id": v_id,
                "values": self.embed(text),
                "metadata": metadata
            })
            
        # Batch upsert
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch, namespace="lead_times")
            
        print(f"✅ Ingested {len(vectors)} lead time records.")

    def ingest_disruptions(self, events: List[Dict[str, Any]]):
        """Embeds and uploads disruption events to the 'disruptions' namespace."""
        if not self.is_ready():
            raise ValueError("VectorStore is not ready. Call init_index() first.")
            
        vectors = []
        for i, event in enumerate(events):
            # Rich text for semantic matching
            text = f"[{event['severity']}] {event['event_type']} at {event['location']}: {event['headline']}. {event['text']}"
            
            metadata = {
                "event_type": event["event_type"],
                "location": event["location"],
                "severity": event["severity"],
                "affected_routes": ", ".join(event["affected_routes"]),
                "date": event["date"],
                "source": event["source"],
                "headline": event["headline"],
                "text": text
            }
            
            v_id = f"dis_{i}_{event['location'].replace(' ', '_')}"
            
            vectors.append({
                "id": v_id,
                "values": self.embed(text),
                "metadata": metadata
            })
            
        self.index.upsert(vectors=vectors, namespace="disruptions")
        print(f"✅ Ingested {len(vectors)} disruption events.")

    def query(self, query_text: str, namespace: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the vector database and returns matching metadata."""
        if not self.is_ready():
            return []
            
        query_vector = self.embed(query_text)
        
        results = self.index.query(
            vector=query_vector,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = []
        for match in results.matches:
            matches.append(match.metadata)
            
        return matches
