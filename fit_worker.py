"""Isolated TabPFN fit+predict for one Model-A config. Run as a subprocess so a
hung API call can be killed by the orchestrator (kill the process, relaunch fresh).

Usage: python fit_worker.py <config_name> <out.pkl>

Trains on a pre-tournament pool (excludes all 2026 WC matches) and predicts every
WC2026 row (group games -> backtest, R32 fixtures -> final). On success writes a
DataFrame [date, home_team, away_team, outcome, p_home_win, p_draw, p_away_win].
"""
import sys
import time
import pandas as pd
import numpy as np

import predict_ensemble as pe  # sets TABPFN_TOKEN, exposes FEATURES
from tabpfn_client import TabPFNClassifier

CUTOFF = pd.Timestamp("2026-06-11")   # WC2026 kickoff; training must predate it
CLASSES = ["home_win", "draw", "away_win"]


def select_pool(pre, name):
    """Pick training rows for a named config. `pre` = all pre-tournament games."""
    if name == "comp1000":                     # recent competitive (no friendlies)
        return pre[pre["importance"] >= 30].tail(1000)
    if name == "major1000":                    # major tournaments / continental / NL
        return pre[pre["importance"] >= 45].tail(1000)
    if name == "recent1500_all":               # everything, most recent
        return pre.tail(1500)
    if name == "wcq800":                        # World Cup + qualifiers only
        return pre[pre["importance"].isin([35.0, 60.0])].tail(800)
    if name == "comp600":                       # smaller competitive
        return pre[pre["importance"] >= 30].tail(600)
    raise ValueError(f"unknown config {name}")


def main():
    name, out = sys.argv[1], sys.argv[2]
    feats = pd.read_pickle("feats_cache.pkl")
    pre = feats[feats["outcome"].notna() & (feats["date"] < CUTOFF)]
    pool = select_pool(pre, name)
    wc = feats[feats["tournament"].eq("FIFA World Cup") &
               feats["date"].dt.year.eq(2026)].copy()
    print(f"[{name}] pool={len(pool)} targets={len(wc)}", flush=True)

    t = time.time()
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[pe.FEATURES].values, pool["outcome"].values)
    proba = clf.predict_proba(wc[pe.FEATURES].values)
    order = {c: i for i, c in enumerate(clf.classes_)}
    P = np.column_stack([proba[:, order[c]] for c in CLASSES])
    print(f"[{name}] fit+predict ok in {time.time()-t:.0f}s", flush=True)

    res = wc[["date", "home_team", "away_team", "outcome"]].copy()
    res["p_home_win"], res["p_draw"], res["p_away_win"] = P[:, 0], P[:, 1], P[:, 2]
    res.to_pickle(out)
    print(f"[{name}] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
