"""Predict an arbitrary set of WC2026 knockout fixtures with the calibrated
TabPFN-led ensemble, on refreshed data (results_live.csv, R32 results merged in).

Same three models and calibration as the R32 pipeline:
  A = TabPFN (comp1000 pool)  B = tournament Poisson  C = Elo expectation
temperature-scaled (T=2.5) and blended 0.5 / 0.2 / 0.3.

Outputs CSV: date,home_team,away_team,p_home_win,p_draw,p_away_win
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from collections import defaultdict

import predict_ensemble as pe
from predict import build_features, importance, FEATURES
from tabpfn_client import TabPFNClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["home_win", "draw", "away_win"]
HOME_ADV = 65.0
CUTOFF = pd.Timestamp("2026-06-11")

# Fixtures to predict (real dataset names). The three locked R16 games plus the
# two expected pairings; neutral knockout venues.
FIXTURES = [
    ("Canada", "Morocco", "2026-07-04"),
    ("Paraguay", "France", "2026-07-04"),
    ("Brazil", "Norway", "2026-07-05"),
    ("Mexico", "England", "2026-07-05"),
    ("United States", "Belgium", "2026-07-06"),
    ("Spain", "Portugal", "2026-07-06"),
    ("Switzerland", "Colombia", "2026-07-07"),
    ("Egypt", "Argentina", "2026-07-07"),   # Egypt beat Australia in R32
]
OUT = os.path.join(HERE, "predictions_R16.csv")


def temp(P, T=2.5):
    Q = np.clip(P, 1e-9, 1) ** (1.0 / T)
    return Q / Q.sum(1, keepdims=True)


def load_base():
    df = pd.read_csv(os.path.join(HERE, "results_live.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["importance"] = df["tournament"].apply(importance)
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["home_win", "away_win"], default="draw")
    df.loc[df["home_score"].isna(), "outcome"] = np.nan
    return df


def poisson_model(df):
    """Tournament Poisson attack/defense from ALL played WC2026 games (group + R32)."""
    wc = df[df["tournament"].eq("FIFA World Cup") & df["date"].dt.year.eq(2026)
            & df["home_score"].notna()]
    ratings = pd.read_csv(os.path.join(pe.DATA_DIR, "team_ratings.csv"))
    inv = {v: k for k, v in pe.NAME_MAP.items()}
    prior = {}                       # keyed by real name
    for _, r in ratings.iterrows():
        prior[pe.m(r["team"])] = r["prior_strength"]
    stat = defaultdict(lambda: dict(P=0, GF=0, GA=0))
    for r in wc.itertuples():
        for t, gf, ga in ((r.home_team, r.home_score, r.away_score),
                          (r.away_team, r.away_score, r.home_score)):
            s = stat[t]; s["P"] += 1; s["GF"] += gf; s["GA"] += ga
    avg = sum(s["GF"] for s in stat.values()) / sum(s["P"] for s in stat.values())

    def att(t): s = stat[t]; return (s["GF"] / s["P"]) / avg if s["P"] else 1.0
    def dfn(t): s = stat[t]; return (s["GA"] / s["P"]) / avg if s["P"] else 1.0
    def pmul(t): return prior.get(t, 65) / 72.0

    def lam(at, df_):
        base = avg * att(at) * dfn(df_)
        return min(4.0, max(0.2, base * (pmul(at) / pmul(df_)) ** 0.5))

    def pois(k, l): return math.exp(-l) * l ** k / math.factorial(k)

    def hda(h, a, mx=10):
        lh, la = lam(h, a), lam(a, h); ph = pd_ = pa = 0.0
        for i in range(mx + 1):
            for j in range(mx + 1):
                p = pois(i, lh) * pois(j, la)
                if i > j: ph += p
                elif i == j: pd_ += p
                else: pa += p
        return [ph, pd_, pa]
    return hda


def elo_hda(he, ae, neutral, dmax=0.32):
    adj = HOME_ADV * (1 - neutral)
    E = 1.0 / (1.0 + 10 ** ((ae - he - adj) / 400.0))
    pd_ = dmax * (1 - abs(2 * E - 1))
    return [E * (1 - pd_), pd_, (1 - E) * (1 - pd_)]


def main():
    df = load_base()
    # append fixtures as unscored knockout rows (neutral venues)
    fx = pd.DataFrame({
        "date": [pd.Timestamp(d) for _, _, d in FIXTURES],
        "home_team": [h for h, _, _ in FIXTURES],
        "away_team": [a for _, a, _ in FIXTURES],
        "tournament": "FIFA World Cup", "neutral": 1,
        "home_score": np.nan, "away_score": np.nan, "outcome": np.nan,
    })
    fx["importance"] = fx["tournament"].apply(importance)
    df = pd.concat([df, fx], ignore_index=True).sort_values("date").reset_index(drop=True)

    feats = build_features(df)
    # locate our fixture rows (unscored, the exact pairs we appended)
    rows = []
    for h, a, _ in FIXTURES:
        m = feats[(feats.home_team == h) & (feats.away_team == a)
                  & feats.home_score.isna()]
        rows.append(m.iloc[-1])
    R = pd.DataFrame(rows)

    # Model A: TabPFN on comp1000 pre-tournament pool
    pool = feats[feats.outcome.notna() & (feats.date < CUTOFF)
                 & (feats.importance >= 30)].tail(1000)
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    pa = clf.predict_proba(R[FEATURES].values)
    order = {c: i for i, c in enumerate(clf.classes_)}
    A = np.column_stack([pa[:, order[c]] for c in CLASSES])

    # Model B: Poisson ; Model C: Elo
    hda = poisson_model(df)
    B = np.array([hda(h, a) for h, a, _ in FIXTURES])
    C = np.array([elo_hda(r.home_elo, r.away_elo, 1) for r in R.itertuples()])

    # calibrate + blend (same params as R32)
    P = 0.5 * temp(A) + 0.2 * temp(B) + 0.3 * temp(C)
    P = P / P.sum(1, keepdims=True)

    out = pd.DataFrame({
        "date": [d for _, _, d in FIXTURES],
        "home_team": [h for h, _, _ in FIXTURES],
        "away_team": [a for _, a, _ in FIXTURES],
        "p_home_win": P[:, 0].round(4),
        "p_draw": P[:, 1].round(4),
        "p_away_win": P[:, 2].round(4),
    })
    out.to_csv(OUT, index=False)
    print("WROTE", OUT, flush=True)
    for r in out.itertuples():
        w = r.home_team if r.p_home_win >= r.p_away_win else r.away_team
        print(f"  {r.home_team:12s} v {r.away_team:12s}  "
              f"H {r.p_home_win:.2f} D {r.p_draw:.2f} A {r.p_away_win:.2f}  -> {w}", flush=True)


if __name__ == "__main__":
    main()
