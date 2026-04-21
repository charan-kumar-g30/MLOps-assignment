import requests
import json

base_url = "http://localhost:8000"

transactions = [
    {"amount": 850.0, "num_transactions_24h": 9, "distance_from_home_km": 88.0, "is_weekend": 1},
    {"amount": 22.0, "num_transactions_24h": 1, "distance_from_home_km": 3.0, "is_weekend": 0},
    {"amount": 430.0, "num_transactions_24h": 4, "distance_from_home_km": 60.0, "is_weekend": 1},
    {"amount": 85.0, "num_transactions_24h": 2, "distance_from_home_km": 12.0, "is_weekend": 0},
    {"amount": 1200.0, "num_transactions_24h": 12, "distance_from_home_km": 110.0, "is_weekend": 1}
]

print("\n--- Sending 5 Test Transactions to /predict ---\n")
print(f"{'Txn':<5} | {'Amount':<8} | {'Txns/24h':<10} | {'Distance':<10} | {'Fraud%':<8} | {'Verdict'}")
print("-" * 65)

for i, txn in enumerate(transactions, 1):
    resp = requests.post(f"{base_url}/predict", json=txn)
    if resp.status_code == 200:
        data = resp.json()
        prob = data['probability'] * 100
        verdict = "FRAUD" if data['is_fraud'] else "legit"
        
        print(f"{i:<5} | ${txn['amount']:<7.0f} | {txn['num_transactions_24h']:<10} | {int(txn['distance_from_home_km'])}km{'':<7} | {prob:.1f}%{'':<3} | {verdict}")
    else:
        print(f"Error for txn {i}: {resp.status_code}")

print("\n--- Testing /predict/batch endpoint ---")
resp = requests.post(f"{base_url}/predict/batch", json=transactions)
if resp.status_code == 200:
    print(f"Batch response OK: returned {len(resp.json())} predictions.")
else:
    print(f"Batch error: {resp.status_code}")

print("\n--- Testing Bonus /metrics endpoint ---")
resp = requests.get(f"{base_url}/metrics")
if resp.status_code == 200:
    print(json.dumps(resp.json(), indent=2))
else:
    print(f"Metrics error: {resp.status_code}")
