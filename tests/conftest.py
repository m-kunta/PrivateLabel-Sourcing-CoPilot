import sys
import os

# Ensure the project root is on sys.path so tests can import scenario_engine, llm_providers, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
