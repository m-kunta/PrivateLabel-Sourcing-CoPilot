# Author: Mohith Kunta
# GitHub: https://github.com/m-kunta

import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_records=50, seed=42):
    """
    Generates realistic private label component lead time data.
    Saves to data/vendor_lead_times.csv.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Base data configuration
    categories = [
        "Apparel/Textiles", 
        "Confectionery/Food", 
        "Wood/Furniture", 
        "Personal Care", 
        "Packaging"
    ]

    components_by_cat = {
        "Apparel/Textiles": ["Cotton", "Spandex Fiber", "Polyester Yarn", "Denim Fabric"],
        "Confectionery/Food": ["Cocoa Butter", "Cocoa Powder", "Palm Oil", "Vanilla Extract"],
        "Wood/Furniture": ["Wood Pulp", "Hardwood Lumber", "MDF", "Veneer"],
        "Personal Care": ["Shea Butter", "Beeswax", "Essential Oils", "Aloe Vera"],
        "Packaging": ["Recycled Cardboard", "Glass Cullet", "PET Flakes", "Plastic Resins"]
    }

    vendors = [
        ("V001", "True Brand Textiles", "True"),
        ("V002", "Frederick's Ingredients", "Frederick's"),
        ("V003", "Casa Home Sourcing", "Casa Home"),
        ("V004", "NatureBest Botanicals", "NatureBest"),
        ("V005", "EcoSource Pack", "EcoSource"),
        ("V006", "Global Threads Co.", "True"),
        ("V007", "Equatorial Ag Supply", "Frederick's"),
        ("V008", "Nordic Wood Products", "Casa Home")
    ]

    routes = [
        {"name": "Trans-Pacific", "origin_country": ["China", "Vietnam", "Indonesia"], "origin_port": ["Shanghai", "Ho Chi Minh", "Jakarta"], "destination_port": ["Los Angeles", "Seattle"], "panama": 0, "suez": 0, "savannah": 0, "west_africa": 0, "hrmz": 0},
        {"name": "Asia-East Coast", "origin_country": ["Bangladesh", "India", "China"], "origin_port": ["Chittagong", "Mumbai", "Shenzhen"], "destination_port": ["Savannah", "New York"], "panama": 1, "suez": 0, "savannah": 1, "west_africa": 0, "hrmz": 1},
        {"name": "West Africa-Atlantic", "origin_country": ["Ghana", "Ivory Coast", "Nigeria"], "origin_port": ["Tema", "Abidjan", "Lagos"], "destination_port": ["New York", "Baltimore"], "panama": 0, "suez": 0, "savannah": 0, "west_africa": 1, "hrmz": 0},
        {"name": "South America-Atlantic", "origin_country": ["Brazil", "Colombia"], "origin_port": ["Santos", "Cartagena"], "destination_port": ["Miami", "Savannah"], "panama": 0, "suez": 0, "savannah": 1, "west_africa": 0, "hrmz": 0},
        {"name": "Europe-East Coast", "origin_country": ["Sweden", "Germany", "Italy"], "origin_port": ["Gothenburg", "Hamburg", "Genoa"], "destination_port": ["Baltimore", "New York"], "panama": 0, "suez": 1, "savannah": 0, "west_africa": 0, "hrmz": 0},
        {"name": "Middle East-Atlantic", "origin_country": ["UAE", "Oman"], "origin_port": ["Jebel Ali", "Salalah"], "destination_port": ["Savannah", "New York"], "panama": 0, "suez": 1, "savannah": 1, "west_africa": 0, "hrmz": 1}
    ]

    records = []
    
    for _ in range(num_records):
        vendor = random.choice(vendors)
        category = random.choice(list(components_by_cat.keys()))
        component = random.choice(components_by_cat[category])
        
        # Pick a logical route based on category
        if category in ["Apparel/Textiles", "Packaging"]:
            route_opts = [r for r in routes if r["name"] in ["Trans-Pacific", "Asia-East Coast", "Middle East-Atlantic"]]
        elif category == "Confectionery/Food":
            route_opts = [r for r in routes if r["name"] in ["West Africa-Atlantic", "South America-Atlantic", "Asia-East Coast"]]
        elif category == "Wood/Furniture":
            route_opts = [r for r in routes if r["name"] in ["Europe-East Coast", "Trans-Pacific", "South America-Atlantic"]]
        else:
            route_opts = routes

        route = random.choice(route_opts)
        origin_idx = random.randint(0, len(route["origin_country"])-1)
        dest_idx = random.randint(0, len(route["destination_port"])-1)
        
        base_lead = random.randint(30, 90)
        variance = round(random.uniform(5.0, 25.0), 1)

        records.append({
            "vendor_id": vendor[0],
            "vendor_name": vendor[1],
            "brand_name": vendor[2],
            "category": category,
            "component": component,
            "origin_country": route["origin_country"][origin_idx],
            "origin_port": route["origin_port"][origin_idx],
            "destination_port": route["destination_port"][dest_idx],
            "transport_mode": "Ocean Freight",
            "route_name": route["name"],
            "base_lead_days": base_lead,
            "historical_variance_pct": variance,
            "panama_canal_exposure": route["panama"],
            "suez_canal_exposure": route["suez"],
            "savannah_port_exposure": route["savannah"],
            "west_africa_port_exposure": route["west_africa"]
        })

    df = pd.DataFrame(records)
    
    # Ensure data dir exists
    os.makedirs("data", exist_ok=True)
    out_path = "data/vendor_lead_times.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Successfully generated data for {num_records} SKUs (seed={seed}).")
    print(f"💾 Saved to {out_path}")

if __name__ == "__main__":
    generate_synthetic_data()
