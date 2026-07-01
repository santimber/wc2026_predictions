"""Predict international football fixtures with TabPFN on engineered features."""
import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, log_loss
from tabpfn_client import TabPFNClassifier

TODAY = pd.Timestamp.now().normalize()
TRAIN_START = pd.Timestamp("2014-01-01")
MAX_TRAIN = 10000
HOME_ADV = 65.0
DATA = "results.csv"
RAW_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "form5_diff", "form10_diff", "home_form5", "away_form5",
    "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5", "gd10_diff",
    "home_streak", "away_streak", "home_rest", "away_rest",
    "home_played", "away_played",
    "h2h_n", "h2h_home_winrate", "h2h_draw_rate", "h2h_gd",
    "neutral", "importance",
]


def importance(t):
    """Map tournament name to an ELO K-factor weight; higher means bigger rating swings."""
    t = t.lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "confederations" in t:
        return 50.0
    if any(k in t for k in [
        "uefa euro", "copa am", "african cup", "asian cup",
        "gold cup", "nations league", "oceania nations"
        ]):
        return 45.0
    if "qualif" in t:
        return 35.0
    if "friendly" in t:
        return 20.0
    return 30.0


def load_data(refresh=False):
    """Load and lightly clean the results CSV, downloading it if missing or refresh=True."""
    if refresh or not os.path.exists(DATA):
        df = pd.read_csv(RAW_URL)
        df.to_csv(DATA, index=False)
    else:
        df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["home_win", "away_win"], default="draw")
    df.loc[df["home_score"].isna(), "outcome"] = np.nan
    df["importance"] = df["tournament"].apply(importance)
    return df


def build_features(df):
    """One chronological pass: every feature uses only matches before kickoff."""
    elo = defaultdict(lambda: 1500.0)
    res = defaultdict(list)
    last_date, h2h = {}, defaultdict(list)

    def team_feats(team):
        """Return pre-match stats for a team: ELO, form averages, win rate, goal stats, streak, games played.
        Defaults represent a mid-table team with no history."""
        r = res[team]
        if not r:
            return elo[team], 1.3, 1.3, 0.33, 1.0, 1.0, 0.0, 0.0, 0
        last5, last10 = r[-5:], r[-10:]
        # each entry is (points, gf, ga, won); walk back until a non-win to count winning streak
        streak = 0
        for p, *_ in reversed(r):
            if p < 1:
                break
            streak += 1
        return (elo[team],
                np.mean([p for p, *_ in last5]), np.mean([p for p, *_ in last10]),
                np.mean([w for *_, w in last10]),
                np.mean([g for _, g, _, _ in last5]), np.mean([a for _, _, a, _ in last5]),
                np.mean([g - a for _, g, a, _ in last10]), streak, len(r))

    def h2h_feats(home, away):
        """Head-to-head record between the two teams, keyed by sorted pair so order doesn't matter.
        GD is flipped for matches where home was the away side."""
        m = h2h[tuple(sorted((home, away)))]
        if not m:
            return 0, 0.5, 0.25, 0.0
        n = len(m)
        return (n,
                sum(w == home for _, _, w in m) / n,
                sum(w == "draw" for _, _, w in m) / n,
                np.mean([g if h == home else -g for h, g, _ in m]))

    rows = []
    for r in df.itertuples():
        h, a, adj = r.home_team, r.away_team, HOME_ADV * (1 - r.neutral)
        he, hf5, hf10, hwr, hgf, hga, hgd, hstk, hn = team_feats(h)
        ae, af5, af10, awr, agf, aga, agd, astk, an = team_feats(a)
        nm, h2h_wr, h2h_dr, h2h_gd = h2h_feats(h, a)
        rows.append({
            "elo_diff": he + adj - ae, "home_elo": he, "away_elo": ae,
            "form5_diff": hf5 - af5, "form10_diff": hf10 - af10,
            "home_form5": hf5, "away_form5": af5,
            "home_winrate": hwr, "away_winrate": awr,
            "home_gf5": hgf, "away_gf5": agf, "home_ga5": hga, "away_ga5": aga,
            "gd10_diff": hgd - agd, "home_streak": hstk, "away_streak": astk,
            "home_rest": min((r.date - last_date[h]).days, 90) if h in last_date else 30,
            "away_rest": min((r.date - last_date[a]).days, 90) if a in last_date else 30,
            "home_played": hn, "away_played": an,
            "h2h_n": nm, "h2h_home_winrate": h2h_wr, "h2h_draw_rate": h2h_dr, "h2h_gd": h2h_gd,
        })

        if not np.isnan(r.home_score):
            gd = r.home_score - r.away_score
            # standard ELO expected score from home's perspective, with home-advantage baked into adj
            exp = 1 / (1 + 10 ** ((ae - he - adj) / 400))
            s = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
            # goal-difference multiplier (FIFA-style): bigger wins shift ratings more
            g = 1.0 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8)
            delta = r.importance * g * (s - exp)
            elo[h] += delta
            elo[a] -= delta
            res[h].append((3 if gd > 0 else (1 if gd == 0 else 0), r.home_score, r.away_score, gd > 0))
            res[a].append((3 if gd < 0 else (1 if gd == 0 else 0), r.away_score, r.home_score, gd < 0))
            last_date[h] = last_date[a] = r.date
            h2h[tuple(sorted((h, a)))].append((h, gd, h if gd > 0 else (a if gd < 0 else "draw")))

    return df.join(pd.DataFrame(rows, index=df.index))


def train(pool):
    """Fit TabPFN on the feature matrix; ignore_pretraining_limits allows >1000 rows."""
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    return clf


def main():
    """Backtest on the previous calendar month, then predict all upcoming fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download dataset from source")
    args = parser.parse_args()

    df = load_data(refresh=args.refresh)
    latest_date = df[df["date"].notna()]["date"].max()
    print(f"Latest game in dataset: {latest_date.date()}")
    print(f"Data freshness: {pd.Timestamp.now() - latest_date}")

    feats = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    future = feats[feats["home_score"].isna() & (feats["date"] > TODAY)].sort_values("date")

    month = (TODAY.to_period("M") - 1)
    test = played[(played["date"] >= month.start_time) & (played["date"] < (month + 1).start_time)]
    if len(test):
        clf = train(played[played["date"] < month.start_time].tail(MAX_TRAIN))
        proba = clf.predict_proba(test[FEATURES].values)
        pred = clf.classes_[proba.argmax(1)]
        print(f"\nBacktest {month} ({len(test)} matches): "
              f"accuracy {accuracy_score(test['outcome'], pred):.0%}, "
              f"log-loss {log_loss(test['outcome'], proba, labels=clf.classes_):.3f}")

    clf = train(played.tail(MAX_TRAIN))
    proba = clf.predict_proba(future[FEATURES].values)
    cols = {c: proba[:, i] for i, c in enumerate(clf.classes_)}

    out = future[["date", "home_team", "away_team"]].copy()
    out["predicted"] = clf.classes_[proba.argmax(1)]
    out["p_home_win"] = cols["home_win"]
    out["p_draw"] = cols["draw"]
    out["p_away_win"] = cols["away_win"]

    today_str = pd.Timestamp.now().strftime("%Y%m%d")
    filename = f"predictions_{today_str}.csv"
    out.to_csv(filename, index=False)

    print(f"\n{len(out)} fixture predictions -> {filename}\n")
    for r in out.itertuples():
        print(f"  {r.date.date()}  {r.home_team:>20} vs {r.away_team:<20}  "
              f"-> {r.predicted:<9}  H {r.p_home_win:4.0%} | D {r.p_draw:4.0%} | A {r.p_away_win:4.0%}")


if __name__ == "__main__":
    main()
