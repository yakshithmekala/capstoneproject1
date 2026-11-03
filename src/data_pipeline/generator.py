import datetime
import math
import random
import uuid
import numpy as np
import pandas as pd

# Indian Payment Ecosystem Merchant Categories
MERCHANT_CATEGORIES = [
    'upi_p2p', 'crypto_exchange', 'jewelry_gold', 'electronics_mobiles',
    'online_shopping', 'travel_irctc', 'kirana_grocery', 'petrol_pump', 'restaurants_food'
]

DEVICE_TYPES = ['mobile_app', 'web_browser', 'pos_terminal', 'atm']

# High risk categories in Indian financial context
HIGH_RISK_CATEGORIES = ['crypto_exchange', 'upi_p2p', 'jewelry_gold']

# Indian City Coordinates (Lat, Long)
CITY_COORDINATES = [
    (19.0760, 72.8777),  # Mumbai
    (28.6139, 77.2090),  # Delhi
    (12.9716, 77.5946),  # Bengaluru
    (17.3850, 78.4867),  # Hyderabad
    (13.0827, 80.2707),  # Chennai
    (22.5726, 88.3639),  # Kolkata
    (18.5204, 73.8567),  # Pune
    (23.0225, 72.5714),  # Ahmedabad
]


def generate_synthetic_transactions(num_records=10000, fraud_rate=0.035, random_seed=42):
    """
    Generates synthetic financial transaction data tailored for the Indian financial sector
    with INR (₹) amounts, UPI/P2P transfers, and geographical location jumps across Indian cities.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    num_users = max(50, num_records // 20)
    user_ids = [f"USER_{i:04d}" for i in range(1, num_users + 1)]
    
    # Establish baseline user profiles (in INR ₹)
    user_profiles = {}
    for uid in user_ids:
        home_city = random.choice(CITY_COORDINATES)
        user_profiles[uid] = {
            'avg_amount': float(np.random.gamma(shape=3.0, scale=800.0)), # avg ~ ₹2,400 baseline
            'home_lat': home_city[0] + np.random.normal(0, 0.05),
            'home_long': home_city[1] + np.random.normal(0, 0.05),
            'preferred_device': random.choice(DEVICE_TYPES[:2])
        }

    start_time = datetime.datetime.now() - datetime.timedelta(days=30)
    records = []
    
    user_state = {uid: {'last_time': start_time, 'last_lat': user_profiles[uid]['home_lat'], 'last_long': user_profiles[uid]['home_long']} for uid in user_ids}

    for i in range(num_records):
        txn_id = f"TXN-{uuid.uuid4().hex[:10].upper()}"
        uid = random.choice(user_ids)
        profile = user_profiles[uid]
        state = user_state[uid]
        
        time_delta_sec = random.randint(300, 43200)
        curr_time = state['last_time'] + datetime.timedelta(seconds=time_delta_sec)
        
        is_fraud = 1 if (random.random() < fraud_rate) else 0
        
        if is_fraud:
            fraud_pattern = random.choice(['amount_anomaly', 'velocity_spike', 'location_jump', 'high_risk_device'])
            
            if fraud_pattern == 'amount_anomaly':
                # Huge spike in INR (e.g., ₹1,50,000 to ₹8,50,000)
                amount = round(profile['avg_amount'] * random.uniform(15.0, 45.0), 2)
                category = random.choice(HIGH_RISK_CATEGORIES)
                lat = state['last_lat'] + np.random.normal(0, 0.1)
                long = state['last_long'] + np.random.normal(0, 0.1)
                device = profile['preferred_device']
                card_present = 0
            elif fraud_pattern == 'velocity_spike':
                curr_time = state['last_time'] + datetime.timedelta(seconds=random.randint(2, 30))
                amount = round(profile['avg_amount'] * random.uniform(4.0, 10.0), 2)
                category = 'upi_p2p'
                lat = state['last_lat']
                long = state['last_long']
                device = 'mobile_app'
                card_present = 0
            elif fraud_pattern == 'location_jump':
                # Shift to remote Indian city coordinates
                other_city = random.choice([c for c in CITY_COORDINATES if abs(c[0]-profile['home_lat']) > 3])
                lat = other_city[0] + np.random.normal(0, 0.05)
                long = other_city[1] + np.random.normal(0, 0.05)
                curr_time = state['last_time'] + datetime.timedelta(seconds=random.randint(60, 600))
                amount = round(profile['avg_amount'] * random.uniform(8.0, 25.0), 2)
                category = 'jewelry_gold'
                device = 'web_browser'
                card_present = 0
            else: # high_risk_device
                amount = round(profile['avg_amount'] * random.uniform(5.0, 15.0), 2)
                category = random.choice(HIGH_RISK_CATEGORIES)
                lat = state['last_lat']
                long = state['last_long']
                device = 'atm' if profile['preferred_device'] != 'atm' else 'web_browser'
                card_present = 0
        else:
            # Legitimate normal INR transaction (e.g. ₹50 to ₹15,000)
            amount = round(max(20.0, np.random.normal(profile['avg_amount'], profile['avg_amount'] * 0.4)), 2)
            category = random.choice(MERCHANT_CATEGORIES)
            lat = state['last_lat'] + np.random.normal(0, 0.01)
            long = state['last_long'] + np.random.normal(0, 0.01)
            device = profile['preferred_device'] if random.random() < 0.85 else random.choice(DEVICE_TYPES)
            card_present = 1 if device in ['pos_terminal', 'atm'] else 0
            
        ip_prefix = f"{random.randint(49, 183)}.{random.randint(1, 254)}"
        ip_address = f"{ip_prefix}.{random.randint(1, 254)}.{random.randint(1, 254)}"
        
        records.append({
            'transaction_id': txn_id,
            'timestamp': curr_time.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': uid,
            'amount': float(amount),
            'merchant_category': category,
            'device_type': device,
            'location_lat': float(lat),
            'location_long': float(long),
            'ip_address': ip_address,
            'card_present': int(card_present),
            'is_fraud': int(is_fraud)
        })
        
        user_state[uid]['last_time'] = curr_time
        user_state[uid]['last_lat'] = lat
        user_state[uid]['last_long'] = long

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df
