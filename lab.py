"""Autonomous ensemble lab for WC2026 Round-of-32.

Pipeline:
  1. For several Model-A (TabPFN) training-pool configs, fit in an isolated
     subprocess (killable on API hang) and cache leakage-free probabilities for
     every WC2026 game (69 group games with known results + 16 R32 fixtures).
  2. Model B = tournament Poisson: rolling (leakage-free) for the group backtest,
     full-group ratings for the R32 final.
  3. Model C = Elo expectation with a tunable draw spread (leakage-free).
  4. Pick the best Model-A config by group-stage log-loss, then tune ensemble
     weights (wA,wB,wC) + Elo draw spread by 5-fold CV on the 69 group games.
  5. Emit predictions_R32.csv (competition schema), REPORT.md, iteration_log.md.

Resumable: existing per-config caches are reused. Safe to re-run.
"""
import os
import sys
import math
import time
import subprocess
import itertools
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, log_loss

import predict_ensemble as pe

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
CLASSES = ["home_win", "draw", "away_win"]
HOME_ADV = 65.0
CONFIGS = ["comp1000", "major1000", "recent1500_all", "wcq800", "comp600"]
ATTEMPT_TIMEOUT = 230   # seconds per subprocess attempt
ATTEMPTS = 3
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# Model A: run/cached TabPFN config
# ---------------------------------------------------------------------------
def run_config(name):
    """Return cached prediction DataFrame for a config, fitting if needed."""
    out = os.path.join(CACHE, f"A_{name}.pkl")
    if os.path.exists(out):
        log(f"[{name}] cache hit")
        return pd.read_pickle(out)
    for k in range(ATTEMPTS):
        log(f"[{name}] fit attempt {k+1}/{ATTEMPTS}...")
        p = subprocess.Popen([sys.executable, "fit_worker.py", name, out],
                             cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        try:
            # communicate() enforces a hard timeout even during a silent hang
            # (TabPFN's spinner writes '\r', so line-streaming would never fire).
            outp, _ = p.communicate(timeout=ATTEMPT_TIMEOUT)
        except subprocess.TimeoutExpired:
            log(f"[{name}] attempt {k+1} timed out, killing")
            p.kill()
            p.communicate()
            outp = ""
        for line in (outp or "").splitlines():
            line = line.rstrip()
            if line and not any(s in line for s in ("Fitting", "Predicting", "###",
                                "journey", "Report issues", "Press Ctrl",
                                "active development")):
                log(f"    {line}")
        if os.path.exists(out):
            return pd.read_pickle(out)
    log(f"[{name}] FAILED")
    return None


# ---------------------------------------------------------------------------
# Model B: tournament Poisson (rolling for backtest, full for final)
# ---------------------------------------------------------------------------
def poisson_hda(lh, la, mx=10):
    def pois(k, l):
        return math.exp(-l) * l ** k / math.factorial(k)
    ph = pd_ = pa = 0.0
    for i in range(mx + 1):
        for j in range(mx + 1):
            p = pois(i, lh) * pois(j, la)
            if i > j: ph += p
            elif i == j: pd_ += p
            else: pa += p
    return [ph, pd_, pa]


def rolling_poisson(group_df):
    """Leakage-free Poisson for each group game using only prior tournament games.
    group_df: rows with home_team, away_team, date, home_score, away_score (real names)."""
    ratings = pd.read_csv(os.path.join(pe.DATA_DIR, "team_ratings.csv"))
    prior = dict(zip(ratings["team"], ratings["prior_strength"]))
    inv = {v: k for k, v in pe.NAME_MAP.items()}   # real -> data name for prior lookup
    g = group_df.sort_values("date").reset_index()
    stat = defaultdict(lambda: dict(P=0, GF=0, GA=0))
    tot_g = tot_p = 0
    probs = {}
    for r in g.itertuples():
        avg = (tot_g / tot_p) if tot_p else 1.3
        h, a = inv.get(r.home_team, r.home_team), inv.get(r.away_team, r.away_team)

        def att(t):
            s = stat[t]; return (s["GF"] / s["P"]) / avg if s["P"] else 1.0

        def dfn(t):
            s = stat[t]; return (s["GA"] / s["P"]) / avg if s["P"] else 1.0

        def pmul(t):
            return prior.get(t, 65) / 72.0

        def lam(at, df_):
            base = avg * att(at) * dfn(df_)
            p = (pmul(at) / pmul(df_)) ** 0.5
            return min(4.0, max(0.2, base * p))

        probs[(r.home_team, r.away_team)] = poisson_hda(lam(h, a), lam(a, h))
        # now fold this game's result into the running stats
        hg, ag = r.home_score, r.away_score
        for t, gf, ga in ((h, hg, ag), (a, ag, hg)):
            s = stat[t]; s["P"] += 1; s["GF"] += gf; s["GA"] += ga
        tot_g += hg + ag; tot_p += 2
    return probs


# ---------------------------------------------------------------------------
# Model C: Elo expectation with tunable draw spread
# ---------------------------------------------------------------------------
def elo_hda(he, ae, neutral, dmax):
    adj = HOME_ADV * (1 - neutral)
    E = 1.0 / (1.0 + 10 ** ((ae - he - adj) / 400.0))
    pdraw = dmax * (1 - abs(2 * E - 1))
    return [E * (1 - pdraw), pdraw, (1 - E) * (1 - pdraw)]


def elo_probs(rows, dmax):
    return {(r.home_team, r.away_team): elo_hda(r.home_elo, r.away_elo,
            getattr(r, "neutral", 1), dmax) for r in rows.itertuples()}


# ---------------------------------------------------------------------------
# Metrics / weighting helpers
# ---------------------------------------------------------------------------
def ll(y, P):
    P = np.clip(P, 1e-9, 1)
    P = P / P.sum(1, keepdims=True)
    return log_loss(y, P, labels=CLASSES)


def acc(y, P):
    return accuracy_score(y, [CLASSES[i] for i in P.argmax(1)])


UNIF = np.array([1 / 3, 1 / 3, 1 / 3])


def blend(mats, w, lam=0.0):
    """Weighted-average the model matrices, then shrink toward uniform by lam
    (a CV-tuned calibration guard so log-loss can't run away from overconfidence)."""
    P = sum(wi * M for wi, M in zip(w, mats))
    P = P / P.sum(1, keepdims=True)
    P = (1 - lam) * P + lam * UNIF
    return P / P.sum(1, keepdims=True)


def simplex(step=0.1):
    pts = [round(x * step, 3) for x in range(int(1 / step) + 1)]
    return [(a, b, round(1 - a - b, 3)) for a in pts for b in pts
            if 0 <= 1 - a - b <= 1 + 1e-9]


# ---------------------------------------------------------------------------
def main():
    feats = pe.load_feats()
    _, fixtures, actuals = pe.assemble()

    wc = feats[feats["tournament"].eq("FIFA World Cup") & feats["date"].dt.year.eq(2026)]
    group = wc[wc["outcome"].notna()].copy()
    gkeys = list(zip(group["home_team"], group["away_team"]))
    y = group["outcome"].values
    log(f"Backtest set: {len(group)} WC2026 group games "
        f"({dict(pd.Series(y).value_counts())})")

    # ---- Model B (rolling group, full R32) ----
    Bg = rolling_poisson(group)
    Bg_mat = np.array([Bg[k] for k in gkeys])
    Br32 = pe.model_b(fixtures)     # full-group Poisson, aligned to fixtures
    log(f"Model B ready (rolling group + full R32). group log-loss={ll(y, Bg_mat):.3f}")

    # ---- Model C (Elo) group matrices for a few dmax; final chosen later ----
    def Cg_mat(dmax):
        cp = elo_probs(group, dmax)
        return np.array([cp[k] for k in gkeys])
    DMAX = [0.16, 0.20, 0.24, 0.28, 0.32]
    log("Model C (Elo) ready. group log-loss by dmax: " +
        ", ".join(f"{d}:{ll(y, Cg_mat(d)):.3f}" for d in DMAX))

    # ---- Model A configs ----
    A_group, A_r32, config_metric = {}, {}, {}
    for name in CONFIGS:
        df = run_config(name)
        if df is None:
            continue
        pm = {(r.home_team, r.away_team): [r.p_home_win, r.p_draw, r.p_away_win]
              for r in df.itertuples()}
        Ag = np.array([pm[k] for k in gkeys])
        A_group[name] = Ag
        A_r32[name] = np.array([pm[(h, a)] for h, a, _ in fixtures])
        config_metric[name] = (ll(y, Ag), acc(y, Ag))
        log(f"[{name}] Model-A group: log-loss={config_metric[name][0]:.3f} "
            f"acc={config_metric[name][1]:.0%}")

    if not A_group:
        log("No Model-A config succeeded; aborting.")
        _write_logs()
        return

    best_cfg = min(config_metric, key=lambda c: config_metric[c][0])
    log(f"\nBest Model-A config: {best_cfg} "
        f"(log-loss {config_metric[best_cfg][0]:.3f})")

    # ---- Tune (wA,wB,wC,dmax) by 5-fold CV on group games ----
    Ag = A_group[best_cfg]
    rng = np.random.RandomState(0)
    folds = rng.permutation(len(group)) % 5
    LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best = None
    for dmax in DMAX:
        Cg = Cg_mat(dmax)
        mats = [Ag, Bg_mat, Cg]
        for w in simplex(0.1):
            if min(w) < 0:
                continue
            for lam in LAMS:
                cv = []
                for f in range(5):
                    te = folds == f
                    if te.sum() == 0:
                        continue
                    cv.append(ll(y[te], blend([m[te] for m in mats], w, lam)))
                score = float(np.mean(cv))
                if best is None or score < best[0]:
                    best = (score, w, dmax, lam)
    cv_ll, w, dmax, lam = best
    fin = blend([Ag, Bg_mat, Cg_mat(dmax)], w, lam)
    insample, ens_acc = ll(y, fin), acc(y, fin)
    log(f"\nTuned weights A/B/C = {w}, Elo dmax={dmax}, shrink lam={lam}")
    log(f"Ensemble group log-loss: CV={cv_ll:.3f}  in-sample={insample:.3f}  "
        f"acc={ens_acc:.0%}  (uniform baseline log-loss=1.099)")

    # ---- Final R32 predictions ----
    cp = elo_probs(feats[feats['home_score'].isna() &
                         feats['tournament'].eq('FIFA World Cup')], dmax)
    Cr = np.array([cp[(h, a)] for h, a, _ in fixtures])
    Pr = blend([A_r32[best_cfg], Br32, Cr], w, lam)

    out = pd.DataFrame({
        "date": [f[2].date() for f in fixtures],
        "home_team": [f[0] for f in fixtures],
        "away_team": [f[1] for f in fixtures],
        "p_home_win": Pr[:, 0], "p_draw": Pr[:, 1], "p_away_win": Pr[:, 2],
    })
    out["predicted"] = [CLASSES[i] for i in Pr.argmax(1)]
    padv = Pr[:, 0] + 0.5 * Pr[:, 1]
    out["winner"] = np.where(padv >= 0.5, out["home_team"], out["away_team"])
    out["p_advance"] = np.where(padv >= 0.5, padv, 1 - padv)
    out = out.sort_values("date").reset_index(drop=True)

    schema = out[["date", "home_team", "away_team", "predicted",
                  "p_home_win", "p_draw", "p_away_win"]]
    schema.to_csv(os.path.join(HERE, "predictions_R32.csv"), index=False)

    # out-of-sample check on the 2 already-played R32 games
    hits = tot = 0
    checks = []
    for (h, a), act in actuals.items():
        row = out[(out.home_team == h) & (out.away_team == a)]
        if not len(row):
            continue
        row = row.iloc[0]
        side = "home_win" if row.winner == h else "away_win"
        ok = side == act
        hits += ok; tot += 1
        checks.append(f"{h} v {a}: predicted {row.winner} | actual {act} -> "
                      f"{'HIT' if ok else 'MISS'}")

    _write_report(out, best_cfg, w, dmax, lam, config_metric, cv_ll, insample,
                  ens_acc, checks, hits, tot)
    _write_logs()
    log(f"\nWrote predictions_R32.csv, REPORT.md. R32 played-game check: {hits}/{tot}")


def _write_logs():
    with open(os.path.join(HERE, "iteration_log.md"), "w") as f:
        f.write("# Ensemble lab iteration log\n\n```\n" + "\n".join(LOG) + "\n```\n")


def _write_report(out, cfg, w, dmax, lam, cm, cv_ll, insample, ens_acc, checks, hits, tot):
    L = ["# WC2026 Round-of-32 — Ensemble Predictions\n",
         f"**Best Model-A config:** `{cfg}`  |  "
         f"**weights A(TabPFN)/B(Poisson)/C(Elo):** {w}  |  "
         f"**Elo draw spread:** {dmax}  |  **shrink λ:** {lam}\n",
         f"**Group-stage backtest (69 games):** ensemble log-loss "
         f"CV={cv_ll:.3f}, in-sample={insample:.3f}, accuracy={ens_acc:.0%}\n",
         "\n## Model-A config comparison (group log-loss / accuracy)\n"]
    for c, (l, a) in sorted(cm.items(), key=lambda x: x[1][0]):
        L.append(f"- `{c}`: log-loss {l:.3f}, acc {a:.0%}"
                 + ("  ← chosen" if c == cfg else ""))
    L.append("\n## Round-of-32 predictions\n")
    L.append("| Date | Match | H% | D% | A% | Winner | Advance% |")
    L.append("|---|---|--:|--:|--:|---|--:|")
    for r in out.itertuples():
        L.append(f"| {r.date} | {r.home_team} vs {r.away_team} "
                 f"| {r.p_home_win*100:.0f} | {r.p_draw*100:.0f} | {r.p_away_win*100:.0f} "
                 f"| **{r.winner}** | {r.p_advance*100:.0f} |")
    if checks:
        L.append(f"\n## Out-of-sample check (already-played R32): {hits}/{tot}\n")
        for c in checks:
            L.append(f"- {c}")
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
