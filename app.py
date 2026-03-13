# Author: Mohith Kunta
# GitHub: https://github.com/m-kunta

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

from vector_store import VectorStore
from scenario_engine import StrategicAnalystChain
from rss_ingest import get_mock_disruptions

# Page config
st.set_page_config(page_title="PL Sourcing Co-Pilot", page_icon="🚢", layout="wide")
load_dotenv(override=True)

# Data Loading
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/vendor_lead_times.csv")
    except FileNotFoundError:
        st.error("Data not found. Please run 'python data_gen.py' first.")
        return pd.DataFrame()

# Initialize session state
if "vs" not in st.session_state:
    st.session_state["vs"] = VectorStore()
if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None

df = load_data()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("LLM Engine")
    provider = st.selectbox("Provider", ["Anthropic", "OpenAI", "Gemini", "Groq", "Ollama"])
    
    # Defaults
    default_model = "claude-3-7-sonnet-20250219"
    if provider == "OpenAI": default_model = "gpt-4o-mini"
    elif provider == "Gemini": default_model = "gemini-2.5-flash"
    elif provider == "Groq": default_model = "llama-3.3-70b-versatile"
    elif provider == "Ollama": default_model = "llama3.2"
        
    model = st.text_input("Model Name", default_model)
    st.session_state["provider"] = provider
    st.session_state["model"] = model
    
    st.divider()
    
    st.subheader("System Status")
    pinecone_ready = st.session_state["vs"].is_ready()
    
    if pinecone_ready:
        st.success("✅ Pinecone Vector DB: Active")
        st.caption(f"Index: {st.session_state['vs'].index_name}")
    else:
        st.warning("⚠️ Pinecone Vector DB: Offline")
        st.caption("Running in Heuristic Fallback Mode")
        
    if st.button("Recheck Connections"):
        st.session_state["vs"] = VectorStore()
        st.rerun()

# Main UI Tabs
st.title("🚢 Private Label Sourcing Co-Pilot")
tab0, tab1, tab2, tab3 = st.tabs(["❓ How to Use", "📊 Scenario Analyzer", "📈 Risk Dashboard", "🗄️ Data Hub"])

with tab0:
    st.markdown("## The Problem")
    st.markdown(
        """
        Private label buying teams at large retailers source raw materials directly from
        global suppliers under their own brand labels — True brand cotton, Frederick's cocoa
        butter, Casa Home wood pulp. When a disruption hits (a port strike, a canal drought,
        a geopolitical flashpoint), these teams have **no rapid way to answer the critical
        "what-if" question**:

        > *"If Savannah strikes next week, which of our 40+ sourced components are at risk,
        > and by how many days?"*

        The result: out-of-stock penalties estimated at **4–8% of annual category revenue**,
        emergency air freight at **6–10× the ocean rate**, and private label brand credibility
        damage that drives shoppers to national brands.
        """
    )

    st.divider()
    st.markdown("## How This Tool Solves It")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 1️⃣ Institutional Memory")
        st.markdown(
            "Vendor lead times, shipping routes, and route-vulnerability flags are stored "
            "in a **Pinecone vector database** — so the system understands *why* a Panama "
            "Canal drought affects Asia-East Coast apparel but not West Africa cocoa."
        )
    with col_b:
        st.markdown("#### 2️⃣ LLM Reasoning")
        st.markdown(
            "An LLM acting as a **Principal Supply Chain Strategist** applies heuristic "
            "disruption coefficients, traces second-order ripple effects, and "
            "risk-classifies every affected component 🔴 Red / 🟡 Yellow / 🟢 Green."
        )
    with col_c:
        st.markdown("#### 3️⃣ Decision-Ready Briefing")
        st.markdown(
            "The output is an **executive summary, key findings, recommended actions, "
            "and a risk horizon** — formatted for a VP of Merchandising and ready to "
            "share with your buying team in seconds."
        )

    st.divider()
    st.markdown("## Steps to Use")

    with st.expander("🔧 First-Time Setup (run once)", expanded=False):
        st.markdown(
            """
            1. **Generate lead time data** — run `python data_gen.py` in your terminal to create `data/vendor_lead_times.csv`.
            2. **Configure your `.env`** — set `LLM_PROVIDER` and the matching API key (Anthropic, OpenAI, Gemini, Groq, or Ollama). Optionally add `PINECONE_API_KEY` for full RAG mode.
            3. **Ingest into Pinecone** *(optional — skip for fallback mode)*:
               - Go to the **🗄️ Data Hub** tab.
               - Click **Initialize Pinecone Index**, then **Ingest Vendor Lead Times**, then **Ingest Disruptions**.
            """
        )

    with st.expander("📊 Running a Scenario (every day use)", expanded=True):
        st.markdown(
            """
            1. Go to the **📊 Scenario Analyzer** tab.
            2. Type a "what-if" disruption question in the text box — or use one of the **Quick Toggles** on the right.
            3. Click **Analyze Scenario**.
            4. Review the **Strategic Analyst Briefing** (executive summary, key findings, recommended actions).
            5. Scroll down to the **Risk Impact Table** — components are colour-coded 🔴 Red / 🟡 Yellow / 🟢 Green and sortable.
            6. Review **Ripple Effects** to see second-order downstream impacts.
            7. Use the **Download CSV** button to export the risk table for your buying team.
            """
        )

    with st.expander("💡 Example Scenarios to Try", expanded=False):
        st.code(
            '"If the Panama Canal transit capacity drops by 30%, how does that ripple through our furniture component lead times?"\n'
            '"A port strike is declared at Savannah. Which of our private label components are critically delayed?"\n'
            '"West Africa cocoa yield has dropped 25% due to El Niño — how does this affect cocoa butter supply?"\n'
            '"If Red Sea shipping is suspended, which routes are forced around Cape of Good Hope and by how many days?"',
            language=""
        )
        st.info(
            "💡 Use the **🔴 Hormuz Toggle** button in the Scenario Analyzer for an instant 2026 "
            "Strait of Hormuz geopolitical scenario — no typing required."
        )

    st.divider()
    st.caption(
        "Running in **Heuristic Fallback Mode** (no Pinecone key needed) — scenario results are structurally "
        "identical but use `data/vendor_lead_times.csv` directly instead of semantic vector retrieval. "
        "Check the sidebar for system status."
    )

with tab1:
    st.markdown("### What-If Analysis Engine")
    st.write("Generate risk-ranked impact tables using live vector data and LLM reasoning.")
    
    col1, col2 = st.columns([3, 1])
    
    if "scenario_input" not in st.session_state:
        st.session_state["scenario_input"] = ""
        
    with col1:
        question = st.text_area(
            "Enter a disruption scenario:", 
            placeholder="e.g. If the Panama Canal transit capacity drops by 30%, how does that ripple through our textile components?",
            key="scenario_input"
        )
    with col2:
        st.markdown("**Quick Toggles:**")
        
        def set_scenario(text):
            st.session_state["scenario_input"] = text
            
        st.button("⛽ The 'Hormuz' Event Toggle (2026)", on_click=set_scenario, args=("With the escalating 2026 geopolitical tensions, what is the impact on fuel surcharges and transit delays if the Strait of Hormuz is blocked?",))
        st.button("💧 Panama Canal Drought (-50%)", on_click=set_scenario, args=("The Panama Canal has implemented a 50% transit reduction. Show me the impact on our East Coast furniture routing.",))
        st.button("⚓ Savannah Labor Strike", on_click=set_scenario, args=("ILWU dockworkers at the Port of Savannah just declared a strike. Which of our components are critically delayed?",))

    if st.button("Analyze Scenario", type="primary", use_container_width=True):
        # We read from the text_area's value (which is stored in 'scenario_input' or local 'question')
        if not question:
            st.warning("Please enter a scenario.")
        else:
            with st.spinner("🧠 Initializing Strategic Analyst Chain..."):
                chain = StrategicAnalystChain(
                    vector_store=st.session_state["vs"], 
                    provider=st.session_state["provider"], 
                    model=st.session_state["model"]
                )
                try:
                    result = chain.analyze_scenario(question, df)
                    st.session_state["last_analysis"] = result
                except Exception as e:
                    st.error(f"Analysis Failed: {str(e)}")

    # Display Analysis Results
    res = st.session_state.get("last_analysis")
    if res:
        st.divider()
        
        # Badges & Source
        sub_col1, sub_col2 = st.columns([1, 1])
        with sub_col1:
            st.subheader("📑 Strategic Analyst Briefing")
        with sub_col2:
            if res.get("source") == "rag+llm":
                st.info("Source: **Vector DB (RAG) + LLM**")
            else:
                st.warning("Source: **Heuristic Fallback + LLM**")

        # Briefing layout
        briefing = res.get("briefing", {})
        
        with st.expander("Executive Summary", expanded=True):
            st.write(briefing.get("executive_summary", "No summary provided."))
            st.caption(f"**Estimated Risk Horizon:** {briefing.get('risk_horizon', 'Unknown')}")
            
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown("#### Key Findings")
            for kf in briefing.get("key_findings", []):
                st.markdown(f"- {kf}")
        with b_col2:
            st.markdown("#### Recommended Actions")
            for ra in briefing.get("recommended_actions", []):
                st.markdown(f"- {ra}")

        st.markdown("### Risk Impact Table")
        
        risk_table = res.get("risk_table", [])
        if risk_table:
            res_df = pd.DataFrame(risk_table)
            
            # Format Risk Level with colors
            def color_risk(val):
                color = '#ff4b4b' if val == 'Red' else '#ffa421' if val == 'Yellow' else '#21c354'
                return f'color: {color}; font-weight: bold'
                
            styled_df = res_df.style.map(color_risk, subset=['risk_level']).format({
                "disruption_coefficient": "{:.2f}x"
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Download button
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "risk_analysis.csv", "text/csv")
        else:
            st.info("No components identified at risk for this scenario.")

        st.markdown("### Ripple Effects")
        ripples = res.get("ripple_effects", [])
        for rip in ripples:
            st.markdown(f"**{rip.get('primary_disruption')}** ➡️ Affecting **{rip.get('affected_route')}**")
            for di in rip.get("downstream_impacts", []):
                st.caption(f"- {di}")

with tab2:
    st.markdown("### Historical Risk Dashboard")
    st.write("Overview of current base lead times and component variances.")
    
    if not df.empty:
        col3, col4 = st.columns(2)
        
        with col3:
            fig1 = px.scatter(
                df, x="base_lead_days", y="historical_variance_pct", 
                color="category", hover_data=["vendor_name", "component"],
                title="Lead Time Volatility by Category"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        with col4:
            route_counts = df.melt(
                id_vars=['component'], 
                value_vars=['panama_canal_exposure', 'suez_canal_exposure', 'savannah_port_exposure', 'west_africa_port_exposure'],
                var_name='Chokepoint', value_name='Exposed'
            )
            route_counts = route_counts[route_counts['Exposed'] == 1]['Chokepoint'].value_counts().reset_index()
            route_counts.columns = ['Chokepoint', 'Component Count']
            route_counts['Chokepoint'] = route_counts['Chokepoint'].str.replace('_exposure', '').str.replace('_', ' ').str.title()
            
            fig2 = px.bar(route_counts, x='Chokepoint', y='Component Count', title="Portfolio Exposure to Chokepoints", color="Chokepoint")
            st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Vector Database Management")
    st.write("Initialize Pinecone and embed supply chain intelligence.")
    
    st.info("Pinecone API Key is managed securely via the `.env` file.")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        if st.button("🏗️ Initialize Pinecone Index", use_container_width=True):
            if not os.getenv("PINECONE_API_KEY"):
                st.error("PINECONE_API_KEY not found in environment.")
            else:
                try:
                    with st.spinner("Creating index on AWS us-east-1... (Approx 60s)"):
                        st.session_state["vs"].init_index()
                    st.success("Index ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Init failed: {e}")
                    
    with col6:
        if st.button("📥 Ingest Vendor Lead Times", use_container_width=True, disabled=not pinecone_ready):
            try:
                with st.spinner("Embedding 50 rows..."):
                    st.session_state["vs"].ingest_lead_times("data/vendor_lead_times.csv")
                st.success("Lead times ingested into Vector DB.")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
                
    with col7:
        if st.button("📰 Ingest Blank News Disruptions", use_container_width=True, disabled=not pinecone_ready):
            try:
                with st.spinner("Embedding Geopolitical Events..."):
                    st.session_state["vs"].ingest_disruptions(get_mock_disruptions())
                st.success("Disruptions ingested into Vector DB.")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
                
    st.divider()
    st.markdown("##### Current Loaded Raw Data")
    st.dataframe(df.head(10), use_container_width=True)
