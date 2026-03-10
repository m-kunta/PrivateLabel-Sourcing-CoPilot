# Author: Mohith Kunta
# GitHub: https://github.com/m-kunta

import datetime

def get_mock_disruptions():
    """
    Returns a curated list of mock disruption events.
    In a real system, these would be parsed live from RSS feeds or News APIs.
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

if __name__ == "__main__":
    disruptions = get_mock_disruptions()
    for d in disruptions:
        print(f"[{d['severity'].upper()}] {d['headline']}")
