import pandas as pd

df = pd.read_csv("data/raw.csv")

df = df[df["amount"] >= 0]

df.to_csv("data/clean.csv", index=False)

print("Clean data saved!")
