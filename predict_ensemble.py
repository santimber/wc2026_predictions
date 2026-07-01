"""World Cup 2026 Round-of-32 predictions via a 3-model ensemble.

Mixed method:
  Model A  TabPFN on the full real international-results history (~49k matches),
           leakage-free ELO / form / head-to-head features, with the completed
           WC2026 group stage merged in so ratings are current through R32.
  Model B  Tournament Poisson goals model (attack/defense indices vs the
           tournament average, blended with a light pre-tournament prior).
           Reproduces build_dataset.py exactly.
  Model C  TabPFN trained on the 72 WC2026 group matches using tournament-
           specific rating-diff features (attack/defense/PPG/prior/GD deltas).

The three H/D/A probability vectors are blended with fixed weights, then the
draw is split 50/50 to give a knockout winner and an advance probability.

Output: predictions_R32.csv  (schema: date, home_team, away_team, predicted,
p_home_win, p_draw, p_away_win) plus a printed report and a backtest.
"""
import os
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, log_loss

# --- Load TabPFN token from .env (login is also browser-cached) --------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env():
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip())


_load_env()
if os.environ.get("TABPFN_TOKEN"):
    os.environ["TABPFN_TOKEN"] = os.environ["TABPFN_TOKEN"].strip().strip('"')

from tabpfn_client import TabPFNClassifier  # noqa: E402
from predict import build_features, FEATURES, importance, MAX_TRAIN  # noqa: E402

TRAIN_START = pd.Timestamp("2014-01-01")
HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer a repo-local data/ (portable), fall back to the parent project's data/.
DATA_DIR = (os.path.join(HERE, "data") if os.path.isdir(os.path.join(HERE, "data"))
            else os.path.join(ROOT, "data"))
CLASSES = ["home_win", "draw", "away_win"]  # fixed class order for blending

# data/ team name -> results.csv team name
NAME_MAP = {
    "USA": "United States",
    "Curacao": "Curaçao",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}
W_A, W_B, W_C = 0.50, 0.30, 0.20  # ensemble weights


def m(name):
    return NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Data assembly: real history + completed WC2026 group + R32 fixtures
# ---------------------------------------------------------------------------
def load_real():
    df = pd.read_csv(os.path.join(HERE, "results.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["importance"] = df["tournament"].apply(importance)
    return df


def outcome(hg, ag):
    return "home_win" if hg > ag else ("away_win" if hg < ag else "draw")


def assemble():
    """Return (df_for_features, r32_fixtures, r32_actuals).

    df_for_features holds all scored history + WC2026 group results, plus every
    R32 game as an UNSCORED fixture so none of them leak into another's features.
    r32_actuals maps the 2 already-played R32 games to their true outcome.
    """
    df = load_real()
    played = pd.read_csv(os.path.join(DATA_DIR, "matches_played.csv"))

    # 1) Fill the WC2026 group scores that the snapshot was missing.
    grp = played[played["stage"] == "Group"].copy()
    score = {(m(h), m(a)): (hg, ag) for h, a, hg, ag in
             zip(grp["home"], grp["away"], grp["home_goals"], grp["away_goals"])}
    wc = df["tournament"].eq("FIFA World Cup") & df["date"].dt.year.eq(2026)
    for i in df[wc & df["home_score"].isna()].index:
        key = (df.at[i, "home_team"], df.at[i, "away_team"])
        if key in score:
            df.at[i, "home_score"], df.at[i, "away_score"] = score[key]
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["home_win", "away_win"], default="draw")
    df.loc[df["home_score"].isna(), "outcome"] = np.nan

    # 2) Build the 16 R32 fixtures (2 played + 14 upcoming).
    ko = played[played["stage"] == "R32"]
    r32_actuals = {(m(h), m(a)): outcome(hg, ag) for h, a, hg, ag in
                   zip(ko["home"], ko["away"], ko["home_goals"], ko["away_goals"])}
    up = pd.read_csv(os.path.join(DATA_DIR, "upcoming_predictions.csv"))
    fixtures = [(m(h), m(a), pd.Timestamp(d)) for h, a, d in
                zip(up["home"], up["away"], up["date"])]
    # the 2 played games, dated just before the upcoming set
    for h, a in r32_actuals:
        fixtures.append((h, a, pd.Timestamp("2026-06-28")))

    fx = pd.DataFrame({
        "date": [f[2] for f in fixtures],
        "home_team": [f[0] for f in fixtures],
        "away_team": [f[1] for f in fixtures],
        "tournament": "FIFA World Cup", "neutral": 1,
        "home_score": np.nan, "away_score": np.nan, "outcome": np.nan,
    })
    fx["importance"] = fx["tournament"].apply(importance)

    df = pd.concat([df, fx], ignore_index=True).sort_values("date").reset_index(drop=True)
    return df, fixtures, r32_actuals


# ---------------------------------------------------------------------------
# Model A: TabPFN on real international history
# ---------------------------------------------------------------------------
def model_a(feats, r32_rows):
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    # Backtest on the previous real calendar month (Model A credibility).
    month = pd.Timestamp("2026-06-01").to_period("M") - 1  # 2026-05
    test = played[(played["date"] >= month.start_time) &
                  (played["date"] < (month + 1).start_time)]
    backtest = None
    if len(test):
        clf = train_tabpfn(played[played["date"] < month.start_time].tail(MAX_TRAIN),
                           FEATURES, "outcome")
        proba = clf.predict_proba(test[FEATURES].values)
        pred = clf.classes_[proba.argmax(1)]
        backtest = (len(test),
                    accuracy_score(test["outcome"], pred),
                    log_loss(test["outcome"], proba, labels=clf.classes_))

    clf = train_tabpfn(played.tail(MAX_TRAIN), FEATURES, "outcome")
    proba = clf.predict_proba(r32_rows[FEATURES].values)
    order = {c: i for i, c in enumerate(clf.classes_)}
    P = np.column_stack([proba[:, order[c]] for c in CLASSES])
    return P, backtest


def train_tabpfn(pool, features, label):
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[features].values, pool[label].values)
    return clf


# ---------------------------------------------------------------------------
# Model B: tournament Poisson (reproduces build_dataset.py)
# ---------------------------------------------------------------------------
def build_poisson():
    played = pd.read_csv(os.path.join(DATA_DIR, "matches_played.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "team_ratings.csv"))
    prior = dict(zip(ratings["team"], ratings["prior_strength"]))
    stat = defaultdict(lambda: dict(P=0, GF=0, GA=0))
    for h, a, hg, ag in zip(played["home"], played["away"],
                            played["home_goals"], played["away_goals"]):
        for t, gf, ga in ((h, hg, ag), (a, ag, hg)):
            s = stat[t]; s["P"] += 1; s["GF"] += gf; s["GA"] += ga
    avg = sum(s["GF"] for s in stat.values()) / sum(s["P"] for s in stat.values())

    def attack(t):
        s = stat[t]; return (s["GF"] / s["P"]) / avg if s["P"] else 1.0

    def defense(t):
        s = stat[t]; return (s["GA"] / s["P"]) / avg if s["P"] else 1.0

    def prior_mult(t):
        return prior.get(t, 65) / 72.0

    def lam(at, df_):
        base = avg * attack(at) * defense(df_)
        p = (prior_mult(at) / prior_mult(df_)) ** 0.5
        return min(4.0, max(0.2, base * p))

    def pois(k, l):
        return math.exp(-l) * l ** k / math.factorial(k)

    def predict(home, away, mx=10):
        lh, la = lam(home, away), lam(away, home)
        ph = pd_ = pa = 0.0
        for i in range(mx + 1):
            for j in range(mx + 1):
                p = pois(i, lh) * pois(j, la)
                if i > j: ph += p
                elif i == j: pd_ += p
                else: pa += p
        return ph, pd_, pa
    return predict


def model_b(fixtures):
    # fixtures use results.csv names; Poisson keys use data/ names -> invert map.
    inv = {v: k for k, v in NAME_MAP.items()}
    predict = build_poisson()
    rows = []
    for h, a, _ in fixtures:
        ph, pd_, pa = predict(inv.get(h, h), inv.get(a, a))
        rows.append([ph, pd_, pa])
    return np.array(rows)


# ---------------------------------------------------------------------------
# Model C: TabPFN on the 72 group games, tournament rating-diff features
# ---------------------------------------------------------------------------
C_FEATURES = ["d_attack", "d_defense", "d_ppg", "d_prior", "d_pts", "d_gd"]


def _diff_row(rt, h, a):
    H, A = rt[h], rt[a]
    return {
        "d_attack": H["attack_idx"] - A["attack_idx"],
        "d_defense": A["defense_idx"] - H["defense_idx"],  # opponent leakiness
        "d_ppg": H["PPG"] - A["PPG"],
        "d_prior": H["prior_strength"] - A["prior_strength"],
        "d_pts": H["Pts"] - A["Pts"],
        "d_gd": H["GD"] - A["GD"],
    }


def model_c(fixtures):
    ratings = pd.read_csv(os.path.join(DATA_DIR, "team_ratings.csv"))
    rt = {r["team"]: r for _, r in ratings.iterrows()}
    played = pd.read_csv(os.path.join(DATA_DIR, "matches_played.csv"))
    grp = played[played["stage"] == "Group"]

    X, y = [], []
    for h, a, hg, ag in zip(grp["home"], grp["away"],
                            grp["home_goals"], grp["away_goals"]):
        X.append(_diff_row(rt, h, a)); y.append(outcome(hg, ag))
    Xtr = pd.DataFrame(X)[C_FEATURES].values
    clf = TabPFNClassifier(random_state=42)
    clf.fit(Xtr, np.array(y))

    inv = {v: k for k, v in NAME_MAP.items()}
    Xte = pd.DataFrame([_diff_row(rt, inv.get(h, h), inv.get(a, a))
                        for h, a, _ in fixtures])[C_FEATURES].values
    proba = clf.predict_proba(Xte)
    order = {c: i for i, c in enumerate(clf.classes_)}
    return np.column_stack([proba[:, order[c]] if c in order
                            else np.zeros(len(fixtures)) for c in CLASSES])


# ---------------------------------------------------------------------------
# Ensemble + report
# ---------------------------------------------------------------------------
def load_feats(cache=os.path.join(HERE, "feats_cache.pkl")):
    """Build (or load cached) the leakage-free feature matrix over all matches.
    Regenerates from source if the cache is missing, so the repo runs from scratch."""
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    df, _, _ = assemble()
    feats = build_features(df)
    feats.to_pickle(cache)
    return feats


def main():
    print("Assembling data (real history + WC2026 group + R32 fixtures)...")
    df, fixtures, actuals = assemble()
    feats = build_features(df)

    # rows of the 16 R32 fixtures, in fixture order
    is_fx = feats["home_score"].isna() & feats["tournament"].eq("FIFA World Cup") \
        & feats["date"].dt.year.eq(2026)
    fxf = feats[is_fx].copy()
    key_to_idx = {(r.home_team, r.away_team): i for i, r in fxf.iterrows()}
    order_idx = [key_to_idx[(h, a)] for h, a, _ in fixtures]
    r32_rows = feats.loc[order_idx]

    print("Model A: TabPFN on real international history...")
    PA, backtest = model_a(feats, r32_rows)
    print("Model B: tournament Poisson...")
    PB = model_b(fixtures)
    print("Model C: TabPFN on WC2026 group form...")
    PC = model_c(fixtures)

    P = W_A * PA + W_B * PB + W_C * PC
    P = P / P.sum(1, keepdims=True)

    out = pd.DataFrame({
        "date": [f[2].date() for f in fixtures],
        "home_team": [f[0] for f in fixtures],
        "away_team": [f[1] for f in fixtures],
        "p_home_win": P[:, 0], "p_draw": P[:, 1], "p_away_win": P[:, 2],
    })
    out["predicted"] = [CLASSES[i] for i in P.argmax(1)]
    # knockout: split the draw, pick a winner + advance probability
    p_home_adv = P[:, 0] + 0.5 * P[:, 1]
    out["winner"] = np.where(p_home_adv >= 0.5, out["home_team"], out["away_team"])
    out["p_winner_advance"] = np.where(p_home_adv >= 0.5, p_home_adv, 1 - p_home_adv)
    out = out.sort_values("date").reset_index(drop=True)

    schema = out[["date", "home_team", "away_team", "predicted",
                  "p_home_win", "p_draw", "p_away_win"]]
    path = os.path.join(HERE, "predictions_R32.csv")
    schema.to_csv(path, index=False)

    # ---- report ----
    if backtest:
        n, acc, ll = backtest
        print(f"\nModel A backtest (real 2026-05, {n} matches): "
              f"accuracy {acc:.0%}, log-loss {ll:.3f}")

    print(f"\nAll 16 Round-of-32 predictions  ->  {path}\n")
    print(f"{'Match':<34}{'Win%H':>7}{'Draw%':>7}{'Win%A':>7}  {'-> WINNER (advance %)':<28}")
    print("-" * 92)
    for r in out.itertuples():
        tag = ""
        act = actuals.get((r.home_team, r.away_team))
        if act:
            hit = "OK" if ((act == "home_win" and r.winner == r.home_team) or
                           (act == "away_win" and r.winner == r.away_team)) else "MISS"
            tag = f"  [played: {act} -> {hit}]"
        print(f"{r.home_team+' v '+r.away_team:<34}"
              f"{r.p_home_win*100:>6.0f}%{r.p_draw*100:>6.0f}%{r.p_away_win*100:>6.0f}%"
              f"   {r.winner+' ('+format(r.p_winner_advance*100,'.0f')+'%)':<26}{tag}")

    # backtest on the 2 already-played R32 games
    hits = 0
    for (h, a), act in actuals.items():
        row = out[(out.home_team == h) & (out.away_team == a)].iloc[0]
        win_side = "home_win" if row.winner == h else "away_win"
        hits += int(win_side == act)
    print(f"\nOut-of-sample R32 check: {hits}/{len(actuals)} played games called correctly.")


if __name__ == "__main__":
    main()
