"""Deterministic synthetic data for infrastructure checks, never BTS results."""
import random
from datetime import date, timedelta

import pandas as pd

from src.contracts import CLASSES


def synthetic_dataset(size=240, seed=42, start="2022-01-01"):
    rng = random.Random(seed)
    first = date.fromisoformat(start)
    rows, labels = [], []
    routes = [("LIS", "OPO"), ("AAA", "BBB"), ("CCC", "DDD")]
    for i in range(size):
        group = i % 3
        origin, destination = routes[group]
        hour = (6 + 6 * group + rng.randrange(3)) % 24
        rows.append({
            "FL_DATE": (first + timedelta(days=i % 28)).isoformat(),
            "OP_CARRIER_AIRLINE_ID": str(100 + group),
            "ORIGIN": origin, "DEST": destination,
            "CRS_DEP_TIME": f"{hour:02d}30",
            "CRS_ARR_TIME": f"{(hour + 2) % 24:02d}30",
            "DISTANCE": float(200 + group * 700 + rng.randrange(100)),
            "CRS_ELAPSED_TIME": 120.0,
        })
        labels.append(CLASSES[group])
    return pd.DataFrame(rows), pd.Series(labels, name="delay_severity")
