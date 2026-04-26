# Author: Mohith Kunta
# GitHub: https://github.com/m-kunta

import datetime
import json
import logging
import re
import feedparser
from bs4 import BeautifulSoup
from llm_providers import get_llm_response

def get_mock_disruptions():
    """
    Returns a curated list of mock disruption events.
    Used as a fallback when live RSS feeds are unavailable or yield no supply chain disruptions.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    
    return [
        {
            "event_type": "Chokepoint Constraint",
            "location": "Panama Canal",
            "severity": "High",
            "affected_routes": ["Asia-East Coast", "South America-Atlantic"],
            "date": now,
            "source": "Maritime Advisory Board",
            "headline": "Panama Canal Authority Announces 30% Transit Reduction",
            "text": "Due to unprecedented low water levels in Gatun Lake, the Panama Canal Authority has mandated a 30% reduction in daily ship transits. Vessels lacking advanced booking slots face queue times exceeding 14 days. Carriers are considering rerouting Asia-East Coast services via the Suez Canal or Cape of Good Hope, adding significant lead times and fuel surcharges."
        },
        {
            "event_type": "Labor Action",
            "location": "Port of Savannah",
            "severity": "High",
            "affected_routes": ["Asia-East Coast", "Europe-East Coast", "Middle East-Atlantic", "South America-Atlantic"],
            "date": now,
            "source": "Logistics Weekly",
            "headline": "ILWU Dockworkers Initiate Work-to-Rule at Savannah Port",
            "text": "Contract negotiations have stalled, prompting unionized workers at the Port of Savannah to begin a 'work-to-rule' action. Terminal velocity has dropped by 45%, leading to acute vessel bunching. Importers using Savannah as their primary East Coast clearing port should expect dwell times to increase by up to 15-20 days until a resolution is reached."
        },
        {
            "event_type": "Geopolitical Tension",
            "location": "Strait of Hormuz",
            "severity": "Critical",
            "affected_routes": ["Middle East-Atlantic", "Asia-East Coast"],
            "date": now,
            "source": "Global Security Desk",
            "headline": "Escalating 2026 Tensions Threaten Strait of Hormuz Navigation",
            "text": "Rising geopolitical friction in the Middle East has led to naval skirmishes near the Strait of Hormuz. Major shipping alliances have announced a temporary suspension of transits through the region. Vessels must reroute or hold indefinitely. Consequently, global ocean freight rates are expected to see massive fuel surcharge spikes (+15-30%) and routing delays ranging from 12 to 24 days."
        },
        {
            "event_type": "Port Congestion",
            "location": "Tema, Ghana",
            "severity": "Medium",
            "affected_routes": ["West Africa-Atlantic"],
            "date": now,
            "source": "West Africa Trade News",
            "headline": "Severe Yard Congestion Paralyzes Tema Port",
            "text": "A sudden influx of scheduled cocoa and palm oil exports mixed with terminal software outages has caused severe yard congestion at Tema Port in Ghana. Ships are anchoring offshore for an average of 5 to 8 days before securing a berth. Agricultural exports, particularly cocoa butter, are directly impacted."
        },
        {
            "event_type": "Natural Disaster",
            "location": "Chittagong, Bangladesh",
            "severity": "Medium",
            "affected_routes": ["Asia-East Coast", "Trans-Pacific"],
            "date": now,
            "source": "Reuters Supply Chain",
            "headline": "Monsoon Flooding Severs Chittagong Inland Corridors",
            "text": "Unseasonal monsoon flooding has washed out major highways connecting inland garment factories in Bangladesh to the Port of Chittagong. While the port operations remain functional, the inability to move finished textiles and cotton goods from factory to port is creating an effective 7-12 day gap in the outbound supply chain."
        },
        {
            "event_type": "Geopolitical Tension",
            "location": "Red Sea / Suez Canal",
            "severity": "High",
            "affected_routes": ["Europe-East Coast", "Asia-East Coast", "Middle East-Atlantic"],
            "date": now,
            "source": "FreightWaves",
            "headline": "Carriers Maintain Cape Detours Amid Ongoing Red Sea Security Risks",
            "text": "Major ocean carriers are maintaining their reroutes around the Cape of Good Hope as the security situation in the Red Sea remains volatile. Avoiding the Suez Canal adds approximately 4,000 nautical miles and up to 14 days of transit time for shipments moving between Asia/Middle East and Europe or the US East Coast. Capacity constraints persist."
        }
    ]

def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)

def parse_disruption_with_llm(headline: str, text: str, source: str, provider: str, model: str) -> dict:
    system_prompt = """
    You are an AI supply chain analyst. 
    Analyze the following news item and extract disruption details as JSON.
    If the news item does NOT describe a supply chain disruption, return {"is_disruption": false}.
    If it DOES describe a disruption, return JSON strictly matching this schema:
    {
        "is_disruption": true,
        "event_type": "<e.g., Port Congestion, Labor Action, Geopolitical Tension, Natural Disaster, Chokepoint Constraint>",
        "location": "<Where it is happening>",
        "severity": "<Low, Medium, High, or Critical>",
        "affected_routes": ["<e.g., Asia-East Coast, Trans-Pacific, Europe-East Coast>"],
        "headline": "<Original headline>",
        "text": "<Summary of disruption>"
    }
    Return ONLY valid JSON.
    """
    
    prompt = f"Source: {source}\nHeadline: {headline}\nBody: {text}"
    
    try:
        response = get_llm_response(prompt=prompt, provider=provider, model=model, system_prompt=system_prompt)
        json_str = response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
        parsed = json.loads(json_str)
        return parsed
    except Exception as e:
        logging.error(f"Error parsing disruption with LLM: {e}")
        return {"is_disruption": False}

def get_live_disruptions(provider="Anthropic", model="claude-sonnet-4-6", max_items=5):
    feeds = [
        {"url": "https://www.supplychaindive.com/feeds/news/", "source": "Supply Chain Dive"},
        {"url": "https://www.logisticsmgmt.com/rss", "source": "Logistics Management"}
    ]
    
    disruptions = []
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    for feed_info in feeds:
        try:
            parsed_feed = feedparser.parse(feed_info["url"])
            if not parsed_feed.entries:
                continue
                
            for entry in parsed_feed.entries[:max_items]:
                headline = entry.get("title", "")
                summary_html = entry.get("summary", "") or entry.get("description", "")
                text = clean_html(summary_html)
                
                # Analyze with LLM
                result = parse_disruption_with_llm(headline, text, feed_info["source"], provider, model)
                
                if result.get("is_disruption"):
                    result.pop("is_disruption")
                    result["date"] = today
                    result["source"] = feed_info["source"]
                    if "headline" not in result or not result["headline"]:
                        result["headline"] = headline
                    if "text" not in result or not result["text"]:
                        result["text"] = text
                        
                    disruptions.append(result)
                    
        except Exception as e:
            logging.error(f"Error fetching feed {feed_info['url']}: {e}")
            
    if not disruptions:
        logging.warning("No live disruptions found or parsed. Falling back to mock data.")
        return get_mock_disruptions()
        
    return disruptions

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    provider = os.getenv("LLM_PROVIDER", "Anthropic")
    print(f"Fetching live disruptions using {provider}...")
    
    # Can configure to use whatever model is preferred for quick extraction
    # Defaulting to a faster/cheaper model if Groq or Gemini is available
    if provider == "Anthropic":
        model = "claude-sonnet-4-6"
    elif provider == "Groq":
        model = "llama-3.3-70b-versatile"
    elif provider == "Gemini":
        model = "gemini-2.5-flash"
    elif provider == "OpenAI":
        model = "gpt-4o-mini"
    else:
        model = "llama3.2" # Ollama
        
    disruptions = get_live_disruptions(provider=provider, model=model)
    
    print(f"\nFound {len(disruptions)} disruptions:")
    for d in disruptions:
        print(f"[{d['severity'].upper()}] {d.get('headline', 'No headline')}")
        print(f"  Route: {', '.join(d.get('affected_routes', []))}")
        print(f"  Location: {d.get('location', 'Unknown')}\n")
