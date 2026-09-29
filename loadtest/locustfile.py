"""Locust load test for POST /score.

Hammers the API with realistic single-transaction scoring requests at
fixed concurrency (no think time — saturation profile). Bodies vary in
amount and card attributes and stay within the API's validated schema;
TransactionID is unique per request so persistence never collides, and
TransactionDT stays in the validation range (sim days 120-149) so the
synthetic predictions land OUTSIDE the drift monitor's trailing window
anchored at the replay clock — load traffic must never pollute drift
monitoring.

Run headless through the Makefile:

    make loadtest USERS=50 RUN_TIME=2m

Results land in reports/figures/locust_*.csv (p50/p95/p99 + throughput).
"""

from __future__ import annotations

import itertools
import random

from locust import HttpUser, constant, task

_ids = itertools.count(1_000_000)

CARDS = [
    {"card1": 10000, "card2": 200, "card3": 150, "card4": "visa", "card5": 100, "card6": "debit"},
    {
        "card1": 13550,
        "card2": 375,
        "card3": 185,
        "card4": "mastercard",
        "card5": 224,
        "card6": "credit",
    },
    {"card1": 15066, "card2": 475, "card3": 166, "card4": "visa", "card5": 102, "card6": "credit"},
]

IDENTITY = [
    {},
    {"DeviceInfo": "Windows", "id_30": "Windows 10", "id_31": "chrome 62.0", "id_33": "1920x1080"},
    {"DeviceInfo": "iOS Device", "id_30": "iOS 11.2.1", "id_31": "safari mobile 11.0"},
]


def body() -> dict:
    base = {
        "TransactionID": next(_ids),
        "TransactionDT": 10_368_000 + random.randrange(0, 2_592_000),
        "TransactionAmt": round(random.lognormvariate(3.8, 1.1), 2),
        "addr1": random.choice([100, 204, 330, 423]),
        "addr2": 87,
        "dist1": round(random.expovariate(1 / 20), 1),
        "P_emaildomain": random.choice(["gmail.com", "hotmail.com", "anonymous.com", None]),
        "C1": random.randrange(0, 12),
        "C13": random.randrange(0, 20),
        "D1": round(random.uniform(0, 60), 1),
    }
    base.update(random.choice(CARDS))
    base.update(random.choice(IDENTITY))
    return base


class ScoringUser(HttpUser):
    wait_time = constant(0)  # saturation: each virtual user fires continuously
    host = "http://localhost:8000"

    @task
    def score(self) -> None:
        self.client.post("/score", json=body(), name="POST /score")
