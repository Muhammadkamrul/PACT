"""
intact/mild/features.py
=======================
Feature engineering for MILD, ADAPTED FROM THE ORIGINAL to INTACT's context.

WHAT CHANGED FROM THE ORIGINAL MILD, AND WHY
--------------------------------------------
Original MILD features were datacentre/service telemetry:
    cpu_pct, ram_pct, sri, snet, api_latency, analytics_tput, telemetry_queue
Those do not exist in a RAN.  INTACT's risk predictor must read the same
quantities the rest of the framework reasons about, or the two halves of the
system will disagree about what "at risk" means.

So the feature set becomes, per timestep:

  PER-INTENT (the most important block -- MILD's original features had no
  equivalent, because it had no notion of a normalised intent margin):
    g_i             the dimensionless margin              <- v11 §4
    rho_i           rolling fulfilment fraction           <- v11 §3.3
    slack_i         g_i normalised by its own volatility
  PER-TENANT:
    throughput, delay, buffer occupancy
  CELL:
    PRB utilisation, retransmission PRBs, TX power
  CONTROL STATE:
    the current value of every knob (normalised to its domain)

Then the SAME rolling transform the original used -- mean and std over two
window lengths -- because that is what lets the model see a TRAJECTORY
rather than an instant.  That is exactly what the v11 §10.2 ratchet argument
requires: an intent whose margin is comfortable but FALLING must raise
p_hat before it crosses zero.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# the rolling windows kept from the original MILD feature engineering
ROLL_WINDOWS = (5, 15)


def raw_frame(records: List[Dict], intents, tenants, controls: List[str]) -> pd.DataFrame:
    """
    Turn a list of per-slot observations into a flat dataframe.

    Each record is:
        {"t": int, "g": {iid: float}, "rho": {iid: float},
         "kpm": {tenant: {...}, "_cell": {...}}, "controls": {param: value}}
    """
    rows = []
    for r in records:
        row = {"t": r["t"]}
        for iid in intents:
            row[f"g_{iid}"] = r["g"].get(iid, 0.0)
            row[f"rho_{iid}"] = r["rho"].get(iid, 1.0)
        for tid in tenants:
            k = r["kpm"].get(tid, {})
            row[f"tput_{tid}"] = k.get("throughput_mbps", 0.0)
            row[f"delay_{tid}"] = k.get("delay_ms", 0.0)
            row[f"buf_{tid}"] = k.get("buffer_kb", 0.0)
            # Exogenous offered-input forecast from the traffic/RIC layer.
            # This is not a future KPI oracle: it contains only demand known
            # from a schedule/forecast, never future fading or actions.
            row[f"load_{tid}"] = k.get("offered_input_ratio", 1.0)
            row[f"load_forecast_{tid}"] = k.get(
                "offered_input_ratio_forecast", row[f"load_{tid}"])
            row[f"load_forecast_mean_{tid}"] = k.get(
                "offered_input_ratio_forecast_mean", row[f"load_forecast_{tid}"])
            row[f"load_forecast_peak_{tid}"] = k.get(
                "offered_input_ratio_forecast_peak", row[f"load_forecast_{tid}"])
            row[f"load_forecast_min_{tid}"] = k.get(
                "offered_input_ratio_forecast_min", row[f"load_forecast_{tid}"])
        c = r["kpm"]["_cell"]
        row["cell_util"] = c.get("prb_util_pct", 0.0)
        row["cell_retx"] = c.get("retx_prb", 0.0)
        row["cell_txpower"] = c.get("txpower_dbm", 43.0)
        row["cell_offered_input"] = c.get("offered_input_ratio", 1.0)
        row["cell_offered_forecast"] = c.get(
            "offered_input_ratio_forecast", row["cell_offered_input"])
        row["cell_offered_forecast_mean"] = c.get(
            "offered_input_ratio_forecast_mean", row["cell_offered_forecast"])
        row["cell_offered_forecast_peak"] = c.get(
            "offered_input_ratio_forecast_peak", row["cell_offered_forecast"])
        row["cell_offered_forecast_min"] = c.get(
            "offered_input_ratio_forecast_min", row["cell_offered_forecast"])
        row["cell_budget_scale"] = c.get("cell_budget_scale", 1.0)
        for p in controls:
            row[f"ctl_{p}"] = r["controls"].get(p, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def engineer(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Rolling mean/std over two windows -- kept verbatim in SPIRIT from the
    original `_engineer`, but applied to the RAN/intent columns above.

    Plus one addition the original did not have and INTACT needs: an explicit
    DELTA column per margin.  The ratchet failure mode is a slow monotone
    erosion, and a first difference is the cheapest possible way to make that
    visible to the model.
    """
    d = df.copy()
    base = [c for c in d.columns if c != "t"]
    roll_cols = [c for c in base
                 if c.startswith(("g_", "rho_", "tput_", "delay_", "buf_",
                                  "load_", "cell_"))]
    derived = {}
    for c in roll_cols:
        for w in ROLL_WINDOWS:
            derived[f"{c}_mean_{w}"] = d[c].rolling(w, min_periods=1).mean()
            derived[f"{c}_std_{w}"] = d[c].rolling(w, min_periods=1).std()
    # first differences on the margins: the ratchet detector
    for c in [c for c in base if c.startswith("g_")]:
        derived[f"{c}_d1"] = d[c].diff()
        derived[f"{c}_d5"] = d[c].diff(5)
    if derived:
        d = pd.concat([d, pd.DataFrame(derived, index=d.index)], axis=1)
    d = d.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    feats = [c for c in d.columns if c != "t"]
    return d[feats].values.astype("float32"), feats


class Standardiser:
    """Zero-mean unit-variance, fitted on TRAIN ONLY.  Saved with the model
    so inference at runtime uses exactly the training statistics."""

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray) -> "Standardiser":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def fit_transform(self, X):
        return self.fit(X).transform(X)
