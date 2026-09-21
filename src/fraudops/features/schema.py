"""Feature schema: which raw columns enter the model, and how.

Kept deliberately small and documented (engineering clarity over leaderboard
performance):

- ID-like and label-like columns (card1..6, addr1/2, domains, M-flags,
  device fields) are treated as categoricals — LightGBM consumes pandas
  ``category`` columns natively.
- C1-C14 / D1-D15 and the numeric identity columns stay numeric (float32);
  missing values are left as NaN, which LightGBM handles natively.
- Three engineered features: log TransactionAmt, simulated hour-of-day and
  simulated day-of-week. The absolute simulated day is intentionally NOT a
  feature: a tree cannot extrapolate a time trend beyond the training window,
  and the model must score the later stream.
- The anonymised V-block is excluded from the baseline (toggle in
  configs/model.yaml). No competition-style group-key aggregation features.
"""

from __future__ import annotations

CATEGORICAL_TX = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "P_emaildomain",
    "R_emaildomain",
]

NUMERIC_TX = [
    "TransactionAmt",
    "dist1",
    "dist2",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
]

CATEGORICAL_ID = [
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]

NUMERIC_ID = [
    "id_01",
    "id_02",
    "id_03",
    "id_04",
    "id_05",
    "id_06",
    "id_07",
    "id_08",
    "id_09",
    "id_10",
    "id_11",
    "id_13",
    "id_14",
    "id_17",
    "id_18",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_24",
    "id_25",
    "id_26",
    "id_32",
]

# Engineered features derived inside the pipeline (never raw inputs).
ENGINEERED = ["log_TransactionAmt", "sim_hour", "sim_dow"]

FLAG_COLUMNS = ["has_identity"]

V_PREFIX = "V"
