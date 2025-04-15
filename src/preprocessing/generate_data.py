import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Create sample clickstream data
def generate_clickstream_data(n_records=10000):
    now = datetime.now()
    
    # Generate user IDs (some will be repeated to simulate return visitors)
    user_ids = [f"user_{i}" for i in range(1, 501)]
    
    # Create page categories and specific pages
    page_categories = ["home", "product", "category", "cart", "checkout", "payment", "confirmation"]
    pages = {
        "home": ["/home", "/"],
        "product": [f"/product/{i}" for i in range(1, 101)],
        "category": [f"/category/{cat}" for cat in ["electronics", "clothing", "home", "sports", "beauty"]],
        "cart": ["/cart"],
        "checkout": ["/checkout"],
        "payment": ["/payment"],
        "confirmation": ["/confirmation"]
    }
    
    # Generate possible events
    events = ["page_view", "click", "add_to_cart", "remove_from_cart", "purchase"]
    
    # Device types
    devices = ["desktop", "mobile", "tablet"]
    
    # Referrer sources
    referrers = ["direct", "google", "facebook", "twitter", "email", "organic"]
    
    data = []
    for _ in range(n_records):
        user_id = random.choice(user_ids)
        timestamp = (now - timedelta(minutes=random.randint(0, 10080))).strftime("%Y-%m-%d %H:%M:%S")
        category = random.choice(page_categories)
        page = random.choice(pages[category])
        event = random.choice(events)
        device = random.choice(devices)
        referrer = random.choice(referrers)
        session_id = f"session_{random.randint(1, 1000)}"
        
        # Add some features to make the dataset more realistic
        time_spent = np.random.exponential(60) if event == "page_view" else 0
        if event == "page_view":
            time_spent = round(np.random.exponential(60), 2)
        elif event == "click":
            time_spent = round(np.random.exponential(5), 2)
        else:
            time_spent = round(np.random.exponential(20), 2)
            
        # IP address
        ip = f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
        
        # Browser
        browsers = ["Chrome", "Firefox", "Safari", "Edge"]
        browser = random.choice(browsers)
        
        data.append({
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "page_category": category,
            "page_url": page,
            "event_type": event,
            "device_type": device,
            "referrer": referrer,
            "time_spent": time_spent,
            "ip_address": ip,
            "browser": browser
        })
    
    df = pd.DataFrame(data)
    return df

# Generate and save the dataset
clickstream_df = generate_clickstream_data(50000)
clickstream_df.to_csv("data/raw/clickstream_data.csv", index=False)
print(f"Dataset created with {len(clickstream_df)} records")