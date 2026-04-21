import os

import numpy as np
import pandas as pd


def main() -> None:
    os.makedirs("data", exist_ok=True)

    rng = np.random.default_rng(43)
    row_count = 2000

    data = pd.DataFrame(
        {
            "transaction_id": range(row_count),
            "amount": rng.normal(loc=250, scale=200, size=row_count).round(2),
            "is_fraud": rng.choice([0, 1], size=row_count, p=[0.95, 0.05]),
        }
    )

    data.to_csv("data/raw.csv", index=False)
    print("Raw data created!")


if __name__ == "__main__":
    main()
